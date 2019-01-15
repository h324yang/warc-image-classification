# warc-image-classification


#### Downliading and Compiling AUT
    $ git clone https://github.com/archivesunleashed/aut.git
    $ cd aut
    $ mvn clean install -DskipTests

#### Downloading Spark 
	$ curl -L "https://archive.apache.org/dist/spark/spark-2.3.2/spark-2.3.2-bin-hadoop2.7.tgz" > spark-2.3.2-bin-hadoop2.7.tgz 
	$ tar -xvf spark-2.3.2-bin-hadoop2.7.tgz

#### Installing Dependency
	$ pip install -r req.txt

#### Downloading Pretrained Model
    $ curl -L "https://drive.google.com/uc?export=download&id=1rehRyU_EhvF796Gf7bUWjWpfn8fUwExT" > model/pokedex/pokedex.model

#### Run
	$ python3 predict.py --warc [path]
> A [path] example: /tuna1/scratch/nruest/geocites/warcs/1/
