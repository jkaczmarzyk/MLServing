#!/bin/bash
curl -X POST -H "Content-Type: application/json" -d '{ "text": "hello world"}' http://localhost:5000/emotion
