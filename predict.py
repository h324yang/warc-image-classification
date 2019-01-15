# GEOCITY = "/tuna1/scratch/nruest/geocites/warcs/1/"
AUT_JAR = "aut/target/aut-0.17.1-SNAPSHOT-fatjar.jar"
SPARK = "spark-2.3.2-bin-hadoop2.7/bin"
AUT_PY = "aut/src/main/python"
LABELLER = "model/pokedex/lb.pickle"
MODEL = "model/pokedex/pokedex.model"
# MASTER = "spark://127.0.1.1:7077"
MASTER = "local"
IMG_SIZE = (96, 96, 3)
CLASSES = 5


from os import listdir
import sys
sys.path.append(AUT_PY)
sys.path.append(SPARK)

from aut.common import WebArchive
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext, DataFrame
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql.types import *
import tensorflow as tf
import keras
from keras import models
from model.pokedex import pokedex
from PIL import Image
import io
import base64
import argparse
import pickle
import numpy as np
import pandas as pd


def init_spark(master, aut):
    conf = SparkConf()
    conf.set("spark.jars", aut)
    conf.set("spark.sql.execution.arrow.enabled", "true")
    conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "12800")
#     conf.set("spark.executor.memory", "1G")
#     conf.set("spark.executor.instances", "4")
#     conf.set("spark.executor.cores", "2")
    conf.set("spark.driver.memory", "100G")
    sc = SparkContext("local", "first app", conf=conf)
    sql_context = SQLContext(sc)
    return conf, sc, sql_context


def str2img(byte_str):
    return Image.open(io.BytesIO(base64.b64decode(bytes(byte_str, 'utf-8'))))


def img2np(byte_str, resize=(96,96,3)):
    image = str2img(byte_str)
    img = np.array(image.convert("RGB").resize(resize[0:2])).astype("float") / 255.0
    img_shape = np.shape(img)

    if len(img_shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img_shape[-1] >= 3:
        img = img[:,:,:3]

    return img.flatten().tolist()


def image2feature(byte_str_batch, resize=(96,96,3)):
    def transform(byte_str):
        try:
            return img2np(byte_str)
        except:
            return None

    return byte_str_batch.map(transform)


def parse_image(image_data, resize=(96,96,3)):
    image = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
    image = tf.reshape(image, resize)
    return image


def labeller(classes, threshold=0.7):
    def classify(probs):
        idx = np.argmax(probs)
        if np.max(probs) >= threshold:
            return classes[idx]
        else:
            return "novel category"

    def classify_batch(probs_batch):
        return probs_batch.map(classify)

    return classify_batch


def predict_batch(image_batch):
    image_batch = np.stack(image_batch)
    batch_size = 16

    cls = pokedex.SmallerVGGNet.build(IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], CLASSES)

    image_input = tf.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE[0]*IMG_SIZE[1]*IMG_SIZE[2]])
    dataset = tf.data.Dataset.from_tensor_slices(image_input)
    dataset = dataset.map(parse_image, num_parallel_calls=16).prefetch(6400)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    images_tensor = iterator.get_next()

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    with tf.Session().as_default() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer, feed_dict={image_input: image_batch})
        cls.set_weights(bc_model_weights.value)
        result = []

        try:
            while True:
                images = sess.run(images_tensor)
                probs = cls.predict_on_batch(images)
                result = result + list(probs)

        except tf.errors.OutOfRangeError:
            pass

    return pd.Series(result)


def get_args():
    p = argparse.ArgumentParser(description="detecting pokemon images via Spark + NN")
    p.add_argument("--warc")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    GEOCITY = args.warc
    conf, sc, sql_context = init_spark(MASTER, AUT_JAR)
    W = models.load_model(MODEL).get_weights()
    bc_model_weights = sc.broadcast(W)
    arc = WebArchive(sc, sql_context, GEOCITY)
    df = DataFrame(arc.loader.extractImages(arc.path), sql_context)
    lb = pickle.loads(open(LABELLER, "rb").read())
    classes = lb.classes_
    labelling = labeller(classes)

    img2np_pd_udf = pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)(image2feature)
    predict_batch_udf = pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)(predict_batch)
    labelling_udf = pandas_udf(StringType(), PandasUDFType.SCALAR)(labelling)

    features = df.select("url", "bytes", img2np_pd_udf(col("bytes")).alias("features"))
    data = features[features['features'].isNotNull()].persist()
    predictions_df = data.select("url", "bytes", predict_batch_udf("features").alias("prediction"))
    predictions_df = predictions_df.withColumn("category", labelling_udf("prediction"))
    predictions_df.show()



