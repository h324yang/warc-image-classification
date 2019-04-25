# warc-image-classification


#### Download and Compile AUT
    git clone https://github.com/archivesunleashed/aut.git
    cd aut
    mvn clean install -DskipTests
    cd ..

#### Download Spark 
	curl -L "https://archive.apache.org/dist/spark/spark-2.3.2/spark-2.3.2-bin-hadoop2.7.tgz" > spark-2.3.2-bin-hadoop2.7.tgz 
	tar -xvf spark-2.3.2-bin-hadoop2.7.tgz

#### Install Dependencies
	pip install -r req.txt

#### Set Up StandAlone Mode
    ./spark-2.3.2-bin-hadoop2.7/start-master.sh
    ./spark-2.3.2-bin-hadoop2.7/start-slave.sh 127.0.1.1:7077

#### Run
    sh run_detection.sh

#### Get Images
    sh get_images.sh
