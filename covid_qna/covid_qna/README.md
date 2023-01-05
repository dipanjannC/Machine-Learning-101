# COVID-19 Q&A deployment source code
[Note: Documentation will be updated in Confluence]


[![docker build status](https://img.shields.io/badge/docker_build-passing-emerald.svg)](#) [![docs status](https://img.shields.io/badge/docs-NA-red.svg)](#) 

[![Website](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://13.92.197.252/) 


## Languages used
![Python 3.8](https://img.shields.io/badge/python-v3.8-blue.svg) (app running on flask-restful backend in development mode) [Please modify the requirements file for gunicorn with nginx proxy if running in production mode]

![Go 1.14](https://img.shields.io/github/go-mod/go-version/gomods/athens.svg) (web test UI using very simple HTML is provided running on Gin backend) [HTML frontend will be replaced by react/flutter later]


#### After cloning the repo please make sure you checkout the proper branch in case your system/VM does not support GPU acceleration.
* __Use the *master* branch if your system is configured to use GPU acceleraation or else please use the *no_gpu* branch__
* The code currently provided is still under development.
* Please make sure to check the files and comments and make necessary changes before building with docker.
* This is the python code for deployment only. Once the indexes are built, the lucene index folder should be updated. This code will NOT build the lucene indexes.

## Steps to be followed for building
* Please download and install docker from [here](https://docs.docker.com/get-docker/).
* clone the repo, checkout the proper branch and navigate inside the directory
```
git clone https://swastik_biswas@bitbucket.org/bridgei2idev/covid_qna.git
git checkout master
cd covid_qna
```
* In order to download the lucene indexed data for covid-19 articles navigate to the data folder inside the app directory and run the jar file for Linux or the exe file for Windows
```
cd app/data/
# for linux
java -jar CovidDataDownloader.jar
# for windows
./CovidDataDownloader.exe
```
* Navigate back to the root directory and run the following commands in sequence in order to build and run the docker images
```
cd ../../
docker-compose build
docker-compose up -d
```
* The first command will build the docker images. It will take sometime (about 20-30mins depending on your internet speed and processing speed.)*
* Once the build is done the second command will run it. In order to access the python service open your web browser and navigate to http://localhost:3000/
