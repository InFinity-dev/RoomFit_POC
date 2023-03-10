-- file name : test.sql
-- pwd : /project_name/app/schema/test.sql
 
CREATE DATABASE ROOMFIT_DB default CHARACTER SET UTF8;
 
use ROOMFIT_DB;

CREATE TABLE ROOMFIT_USER (
    USER_ID BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    USER_NAME VARCHAR(255) NOT NULL,
    PASSWORD VARCHAR(256) NOT NULL
) CHARSET=utf8;

CREATE TABLE ROUTINE_MODEL (
    MODEL_ID BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    MODEL_NAME VARCHAR(128) NOT NULL,
    TOTAL_POSE_CNT  INT NOT NULL,
    TOTAL_TIME  INT NOT NULL,
    THUMBNAIL VARCHAR(255) NOT NULL
) CHARSET=utf8;

CREATE TABLE USER_ROUTINE_MODEL (
    USER_MODEL_ID BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    USER_ID BIGINT UNSIGNED NOT NULL,
    MODEL_ID BIGINT UNSIGNED NOT NULL,
    FOREIGN KEY (USER_ID) REFERENCES ROOMFIT_USER(USER_ID) ON UPDATE CASCADE,
    FOREIGN KEY (MODEL_ID) REFERENCES ROUTINE_MODEL(MODEL_ID) ON UPDATE CASCADE
) CHARSET=utf8;

CREATE TABLE ROUTINE_MODEL_POSE (
    POSE_ID BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    MODEL_ID BIGINT UNSIGNED NOT NULL, 
    SEQ_NUM INT NOT NULL,
    POSE_DUR INT NOT NULL,
    FILE_SOURCE VARCHAR(255) NOT NULL,
    FOREIGN KEY (MODEL_ID) REFERENCES ROUTINE_MODEL(MODEL_ID) ON UPDATE CASCADE
) CHARSET=utf8;

CREATE TABLE USER_LOG (
    USER_LOG_ID BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    USER_ID BIGINT UNSIGNED NOT NULL,
    START_TIME DATETIME NOT NULL,
    END_TIME DATETIME NOT NULL,
    FOREIGN KEY (USER_ID) REFERENCES ROOMFIT_USER(USER_ID) ON UPDATE CASCADE
) CHARSET=utf8;