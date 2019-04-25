export PYSPARK_PYTHON=/home/h324yang/pysk_env/bin/python
export PYSPARK_DRIVER_PYTHON=/home/h324yang/pysk_env/bin/python

python aut/src/main/python/tf/detect.py --web_archive "/tuna1/scratch/nruest/geocites/warcs/1/*" \
    --aut_jar aut/target/aut-0.17.1-SNAPSHOT-fatjar.jar \
    --aut_py aut/src/main/python \
    --spark spark-2.3.2-bin-hadoop2.7/bin \
    --master spark://127.0.1.1:7077 \
    --img_model ssd \
    --filter_size 640 640 \
    --output_path warc_res


