import os
import time
import cv2 
from flask import Flask, render_template, Response, jsonify, send_file, request, redirect, flash, url_for, Blueprint
from flask import current_app as current_app
from werkzeug.utils import secure_filename
from module import dbModule
import base64
from io import BytesIO

import angle_check_guide_test
import extract_key_point_guide
import extract_landmark_for_flask
import diff_extract_visualize_for_flask
import video_slice_check_for_flask
from test_db import test_db

app = Flask(__name__)
sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor

UPLOAD_FOLDER = 'static/target_video/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.register_blueprint(test_db)

db_class = dbModule.Database()

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/test_angle')
def test_angle_get():
    """Video streaming home page."""
    return render_template('test_angle.html')

@app.route('/test_angle_post')
def test_angle_post():
    """angle check guide."""
    
    return Response(angle_check_guide_test.run(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/test_angle_score')
def test_angle_score():
    score = angle_check_guide_test.get_score()
    return jsonify({'value': score})

@app.route('/extracted_pose_guide_img')
def extracted_pose_guide_img():
    """showing extracted guide img."""
    guide_img = extract_key_point_guide.ret_image()

    return Response(guide_img, mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['GET'])
def upload_form():
	return render_template('upload.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    model_name = request.form['model_name']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_video filename: ' + filename)
        flash('Video successfully uploaded and displayed below')

        file_name = file.filename.split(".")[0]
        extract_landmark_for_flask.run()
        diff_extract_visualize_for_flask.run(file_name)
        vid_slice_info = video_slice_check_for_flask.run(file_name)

        total_pose_cnt = len(vid_slice_info)
        total_time = sum(pose[2] for pose in vid_slice_info)
        thumbnail = "./static/target_pose/" + file_name + "/pose_1.jpg"
        sql      = "INSERT INTO ROOMFIT_DB.ROUTINE_MODEL(MODEL_NAME, TOTAL_POSE_CNT, TOTAL_TIME, THUMBNAIL) \
                    VALUES('%s', '%d', '%d', '%s')" % (model_name, total_pose_cnt, total_time, thumbnail)
        inserted_id = db_class.execute(sql)

        seq_num = 0
        for pose in vid_slice_info:
            seq_num += 1
            pose_dur = pose[2]
            file_path = "./static/target_pose/" + file_name + "/pose_" + str(seq_num) + ".jpg"
            sql = "INSERT INTO ROOMFIT_DB.ROUTINE_MODEL_POSE(MODEL_ID, SEQ_NUM, POSE_DUR, FILE_SOURCE) \
                    VALUES('%d', '%d', '%d', '%s')" % (inserted_id, seq_num, pose_dur, file_path)
            db_class.execute(sql)
        
        db_class.commit()

        return render_template('my_model_list.html', filename=filename)

@app.route('/my_model_list')
def my_model_list():
    """my model list page."""
    return render_template('my_model_list.html')

@app.route('/models', methods=['GET'])
def read_articles():
    sql      = "SELECT MODEL_NAME, TOTAL_POSE_CNT, TOTAL_TIME, THUMBNAIL \
                FROM ROOMFIT_DB.ROUTINE_MODEL"
    result      = db_class.executeAll(sql)
    print(result)
    return jsonify({'result': 'success', 'models': result})

# @app.route('/display/<filename>')
# def display_video(filename):
# 	#print('display_video filename: ' + filename)
# 	return redirect(url_for('static', filename='uploads/' + filename), code=301)

def gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture('768x576.avi')

    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()  # import image
        if not ret: #if vid finish repeat
            frame = cv2.VideoCapture("768x576.avi")
            continue
        if ret:  # if there is a frame continue with code
            image = cv2.resize(frame, (0, 0), None, 1, 1)  # resize image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
            fgmask = sub.apply(gray)  # uses the background subtraction
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
            closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
            dilation = cv2.dilate(opening, kernel)
            retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            minarea = 400
            maxarea = 50000
            for i in range(len(contours)):  # cycles through all contours in current frame
                if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)
                    area = cv2.contourArea(contours[i])  # area of contour
                    if minarea < area < maxarea:  # area threshold for contour
                        # calculating centroids of contours
                        cnt = contours[i]
                        M = cv2.moments(cnt)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        # gets bounding points of contour to create rectangle
                        # x,y is top left corner and w,h is width and height
                        x, y, w, h = cv2.boundingRect(cnt)
                        # creates a rectangle around contour
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Prints centroid text in order to double check later on
                        cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,.3, (0, 0, 255), 1)
                        cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, markerSize=8, thickness=3,line_type=cv2.LINE_8)
        #cv2.imshow("countours", image)
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
           break
