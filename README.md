# book_recommendation
This project will recommend related books automatically according to the book cover uploaded by users.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

- Python3.7
  - torch >= 1.1.0
  - torchvision >= 0.3.0
  - numpy >= 1.16.3
  - Flask >= 1.0.3

### Installing

Using virutal environment of Python is highly recommended:

```
$ mkdir your_dir
$ cd your_dir
$ python3.7 -m venv venv
$ source venv/bin/activate
```

install packages required by the project

```
$ pip install -r requirements.txt
```

Setup and start the server locally

```
$ export FLASK_APP=book_recommendation
$ export FLASK_ENV=development
$ flask run
```

## Demo

Please refer to [demo](demo/demo.mp4)

## Deployment

Please refer to [flask app deployment](http://flask.pocoo.org/docs/1.0/tutorial/deploy/)