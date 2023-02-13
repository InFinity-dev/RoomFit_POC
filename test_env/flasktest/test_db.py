import os
import time
import cv2 
from flask import Flask, render_template, Response, jsonify, send_file, request, redirect, flash, url_for, Blueprint
from flask import current_app as current_app
from werkzeug.utils import secure_filename
from module import dbModule

test_db = Blueprint('test_db', __name__, url_prefix='/test_db') # db test

# DB 연동 테스트
@test_db.route('/', methods=['GET'])
def index():
    return render_template('db_test.html',
                            result=None,
                            resultData=None,
                            resultUPDATE=None)
 
 
 
# INSERT 함수 예제
@test_db.route('/insert', methods=['GET'])
def insert():
    db_class = dbModule.Database()
 
    sql      = "INSERT INTO testDB.testTable(test) \
                VALUES('%s')" % ('testData')
    db_class.execute(sql)
    db_class.commit()
 
    return render_template('db_test.html',
                           result='insert is done!',
                           resultData=None,
                           resultUPDATE=None)
 
 
 
# SELECT 함수 예제
@test_db.route('/select', methods=['GET'])
def select():
    db_class = dbModule.Database()
 
    sql      = "SELECT idx, test \
                FROM testDB.testTable"
    row      = db_class.executeAll(sql)
 
    print(row)
 
    return render_template('db_test.html',
                            result=None,
                            resultData=row[0],
                            resultUPDATE=None)
 
 
 
# UPDATE 함수 예제
@test_db.route('/update', methods=['GET'])
def update():
    db_class = dbModule.Database()
 
    sql      = "UPDATE testDB.testTable \
                SET test='%s' \
                WHERE test='testData'" % ('update_Data')
    db_class.execute(sql)    
    db_class.commit()
 
    sql      = "SELECT idx, test \
                FROM testDB.testTable"
    row      = db_class.executeAll(sql)
 
    return render_template('db_test.html',
                            result=None,
                            resultData=None,
                            resultUPDATE=row[0])