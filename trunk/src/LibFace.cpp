/** ===========================================================
 * @file LibFace.cpp
 *
 * This file is a part of libface project
 * <a href="http://libface.sourceforge.net">http://libface.sourceforge.net</a>
 *
 * @date    2010-02-18
 * @brief   Lead Face library class.
 * @section DESCRIPTION
 *
 * @author Copyright (C) 2010 by Alex Jironkin
 *         <a href="alexjironkin at gmail dot com">alexjironkin at gmail dot com</a>
 * @author Copyright (C) 2010 by Aditya Bhatt
 *         <a href="adityabhatt at gmail dot com">adityabhatt at gmail dot com</a>
 * @author Copyright (C) 2010 by Gilles Caulier
 *         <a href="mailto:caulier dot gilles at gmail dot com">caulier dot gilles at gmail dot com</a>
 * @author Copyright (C) 2010 by Marcel Wiesweg
 *         <a href="mailto:marcel dot wiesweg at gmx dot de">marcel dot wiesweg at gmx dot de</a>
 * @author Copyright (C) 2011 by Stephan Pleines <a href="mailto:pleines.stephan@gmail.com">pleines.stephan@gmail.com</a>
 *
 * @section LICENSE
 *
 * This program is free software; you can redistribute it
 * and/or modify it under the terms of the GNU General
 * Public License as published by the Free Software Foundation;
 * either version 2, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * ============================================================ */

// own header
#include "LibFace.h"

// LibFace headers
#include "Log.h"
#include "Eigenfaces.h"
#include "FisherFaces.h"
#include "Face.h"
#include "FaceDetect.h"
#include "LibFaceUtils.h"

// OpenCV headers
#if defined (__APPLE__)
#include <highgui.h>
#else
#include <opencv/highgui.h>
#endif

/**
  * New Addition
  */
#include "opencv2/core/core.hpp"
using namespace cv;

using namespace std;

namespace libface {

class LibFace::LibFacePriv {

public:

    /**
     * Constructor.
     *
     * @param argType Type of the face recognition/detection/both to use. ALL by default.
     * @param configDir Config directory of the libface library. If there is a libface.xml, the library will try to load it. Empty ("") by default.
     * @param cascadeDir Directory where haar cascade is. By default it is OPENCVDIR/haarcascades.
     */
    LibFacePriv(Mode argType, const string& argConfigDir, const string& argCascadeDir);

    /**
     * Copy constructor.
     *
     * @param that Object to be copied.
     */
    LibFacePriv(const LibFacePriv& that);

    /**
     * Assignment operator.
     *
     * @param that Object to be copied.
     *
     * @return Reference to assignee.
     */
    LibFacePriv& operator = (const LibFacePriv& that);

    /**
     * Destructor.
     */
    ~LibFacePriv();

    /**
     * TODO
     *
     * @return int TODO
     */
    static int facesize() {
        return 120;
    }

    Mode                    type;
    string                  cascadeDir;
    LibFaceDetectCore*      detectionCore;
    LibFaceRecognitionCore* recognitionCore;
    IplImage*               lastImage;
    string                  lastFileName;

};

LibFace::LibFacePriv::LibFacePriv(Mode argType, const string& argConfigDir, const string& argCascadeDir) : type(argType), cascadeDir(), detectionCore(0), recognitionCore(0), lastImage(0), lastFileName() {
    // We don't need face recognition if we just want detection, and vice versa.
    // So there is a case for everything.
    switch (type) {
    case DETECT:
        LOG(libfaceDEBUG) << "LibFacePriv(...) : type DETECT";
        cascadeDir = argCascadeDir;
        detectionCore = new FaceDetect(cascadeDir);
        break;
    case EIGEN:
        LOG(libfaceDEBUG) << "LibFacePriv(...) : type EIGEN";
        recognitionCore = new Eigenfaces(argConfigDir);
        break;
    case FISHER:
        LOG(libfaceDEBUG) << "LibFacePriv(...) : type FISHER";
        recognitionCore = new Fisherfaces(argConfigDir);
        break;
    case HMM:
        LOG(libfaceDEBUG) << "LibFacePriv(...) : type HMM";
        LOG(libfaceERROR) << "HMM are not implemented yet! Good try though!";
        break;
    default:    // Initialize both detector and Eigenfaces
        LOG(libfaceDEBUG) << "LibFacePriv(...) : type default";
        cout << "LibFace mode: Default" << endl;
        cascadeDir = argCascadeDir;
        detectionCore = new FaceDetect(cascadeDir);
        recognitionCore = new Eigenfaces(argConfigDir);
        break;
    }
}

LibFace::LibFacePriv::LibFacePriv(const LibFacePriv& that) : type(that.type), cascadeDir(that.cascadeDir), detectionCore(0), recognitionCore(0), lastImage(0), lastFileName(that.lastFileName) {
    // copy lastImage
    if(that.lastImage) {
        lastImage = cvCloneImage(that.lastImage);
    }

    // copy detectionCore - construction of new object due to polymorphism
    if(that.detectionCore) {
        if(dynamic_cast<FaceDetect*>(that.detectionCore)) {
            LOG(libfaceDEBUG) << "LibFacePriv(const LibFacePriv& that) : that.detectionCore is of type FaceDetect*.";
            detectionCore = new FaceDetect(*dynamic_cast<FaceDetect*>(that.detectionCore));
        }
        // If other derived classes are implemented, add more cases here.
        if(detectionCore == 0) {
            LOG(libfaceERROR) << "Unable to copy that.detectionCore";
        }
    }

    // copy recognitionCore - construction of new object due to polymorphism
    if(that.recognitionCore) {
        if(dynamic_cast<Eigenfaces*>(that.recognitionCore)) {
            LOG(libfaceDEBUG) << "LibFacePriv(const LibFacePriv& that) : that.recognitionCore is of type Eigenfaces*.";
            recognitionCore = new Eigenfaces(*dynamic_cast<Eigenfaces*>(that.recognitionCore));
        }
        else if(dynamic_cast<Fisherfaces*>(that.recognitionCore)) {
            LOG(libfaceDEBUG) << "LibFacePriv(const LibFacePriv& that) : that.recognitionCore is of type Fisherfaces*.";
            recognitionCore = new Fisherfaces(*dynamic_cast<Fisherfaces*>(that.recognitionCore));
        }

        // If other derived classes are implemented, add more cases here.
        if(recognitionCore == 0) {
            LOG(libfaceERROR) << "Unable to copy that.recognitionCore.";
        }
    }
}

LibFace::LibFacePriv& LibFace::LibFacePriv::operator = (const LibFacePriv& that) {
    if(this == &that) {
        return *this;
    }

    lastFileName = that.lastFileName;
    type = that.type;
    cascadeDir = that.cascadeDir;

    // release lastImage
    if(lastImage) {
        cvReleaseImage(&lastImage);
    }

    // copy that lastImage
    if(that.lastImage) {
        lastImage = cvCloneImage(that.lastImage);
    }


    if( (detectionCore == 0) && (that.detectionCore != 0) ) {
        LOG(libfaceDEBUG) << "LibFacePriv(const LibFacePriv& that) : You are assigning an instance ob LibFace *with* a detectionCore to an instance *without* a detectionCore. This is absolutely possible, but is it really intended?";
    }

    if( (detectionCore != 0) && (that.detectionCore == 0) ) {
        LOG(libfaceDEBUG) << "LibFacePriv(const LibFacePriv& that) : You are assigning an instance ob LibFace *without* a detectionCore to an instance *with* a detectionCore. This is absolutely possible, but is it really intended?";
    }

    // copy detectionCore - construction of new object due to polymorphism - is there a more elegant solution?
    delete detectionCore;
    detectionCore = 0;
    if(that.detectionCore) {
        if(dynamic_cast<FaceDetect*>(that.detectionCore)) {
            LOG(libfaceDEBUG) << "LibFacePriv& operator = (const LibFacePriv& that) : that.detectionCore is of type FaceDetect*.";
            detectionCore = new FaceDetect(*dynamic_cast<FaceDetect*>(that.detectionCore));
        }
        // If other derived classes are implemented, add more cases here.- is there a more elegant solution?
        if(detectionCore == 0) {
            LOG(libfaceERROR) << "Unable to copy that.detectionCore";
        }
    }

    if( (recognitionCore == 0) && (that.recognitionCore != 0) ) {
        LOG(libfaceDEBUG) << "LibFacePriv(const LibFacePriv& that) : You are assigning an instance ob LibFace *with* a recognitionCore to an instance *without* a recognitionCore. This is absolutely possible, but is it really intended?";
    }

    if( (recognitionCore != 0) && (that.recognitionCore == 0) ) {
        LOG(libfaceDEBUG) << "LibFacePriv(const LibFacePriv& that) : You are assigning an instance ob LibFace *without* a recognitionCore to an instance *with* a recognitionCore. This is absolutely possible, but is it really intended?";
    }

    // copy recognitionCore - construction of new object due to polymorphism - is there a more elegant solution?
    delete recognitionCore;
    recognitionCore = 0;
    if(that.recognitionCore) {

        if(dynamic_cast<Eigenfaces*>(that.recognitionCore)) {
            LOG(libfaceDEBUG) << "LibFacePriv& operator = (const LibFacePriv& that) : that.recognitionCore is of type Eigenfaces*.";
            recognitionCore = new Eigenfaces(*dynamic_cast<Eigenfaces*>(that.recognitionCore));
        }

        else if(dynamic_cast<Fisherfaces*>(that.recognitionCore)) {
            LOG(libfaceDEBUG) << "LibFacePriv& operator = (const LibFacePriv& that) : that.recognitionCore is of type Fisherfaces*.";
            recognitionCore = new Fisherfaces(*dynamic_cast<Fisherfaces*>(that.recognitionCore));
        }

        // If other derived classes are implemented, add more cases here.- is there a more elegant solution?
        if(recognitionCore == 0) {
            LOG(libfaceERROR) << "Unable to copy that.recognitionCore.";
        }
    }

    return *this;
}

LibFace::LibFacePriv::~LibFacePriv() {
    delete detectionCore;
    delete recognitionCore;
    if(lastImage) {
        cvReleaseImage(&lastImage);
    }
}

LibFace::LibFace(Mode type, const string& configDir, const string& cascadeDir) : d(new LibFacePriv(type, configDir, cascadeDir)) {
    LOG(libfaceINFO) << "Cascade directory located in : " << cascadeDir;
}

LibFace::LibFace(const LibFace& that) : d(that.d ? new LibFacePriv(*that.d) : 0) {
    if(!d) {
        LOG(libfaceERROR) << "LibFace(const LibFace& that) : d points to NULL.";
    }
}

LibFace& LibFace::operator = (const LibFace& that) {
    if(this == &that) {
        return *this;
    }
    if( (that.d == 0) || (d == 0) ) {
        LOG(libfaceERROR) << "LibFace::operator = (const LibFace& that) : d or that.d points to NULL.";
    } else {
        *d = *that.d;
    }
    return *this;
}

LibFace::~LibFace() {
    delete d;
}

int LibFace::count() const {
    if(noRecognition()) {
        return 0;
    }
    return d->recognitionCore->count();
}

vector<Face*>* LibFace::detectFaces(const string& filename, int scaleFactor) {
    if(noDetection()) {
        return new vector<Face*>;
    }
    if(filename.length() == 0) {
        LOG(libfaceWARNING) << "No image passed for detection.";
        return 0;
    }
    //Check if image was already loaded once, by checking last loaded filename.
    if (filename != d->lastFileName) {
        d->lastFileName = filename;
        if(d->lastImage) {
            cvReleaseImage(&d->lastImage);
        }
        d->lastImage = cvLoadImage(filename.data(), CV_LOAD_IMAGE_GRAYSCALE);
    }

    return d->detectionCore->detectFaces(d->lastImage);
}

vector<Face*>* LibFace::detectFaces(const char* arr, int width, int height, int step, int depth, int channels, int scaleFactor) {
    if(noDetection()) {
        return new vector<Face*>;
    }
    IplImage* image = LibFaceUtils::charToIplImage(arr, width, height, step, depth, channels);
    return d->detectionCore->detectFaces(image);
}

vector<Face*>* LibFace::detectFaces(const IplImage* image) {
    if(noDetection()) {
        return new vector<Face*>;
    }
    return d->detectionCore->detectFaces(image);
}

map<string,string> LibFace::getConfig() {
    map<string,string> result;

    if(noRecognition()) {
        return result;
    }

    result = d->recognitionCore->getConfig();
    return result;
}

double LibFace::getDetectionAccuracy() const {
    if(noDetection()) {
        return 0.0;
    }
    return d->detectionCore->accuracy();
}

int LibFace::getRecommendedImageSizeForDetection(const CvSize&) const {
    return FaceDetect::getRecommendedImageSizeForDetection();
}

CvSize LibFace::getRecommendedImageSizeForRecognition(const CvSize&) const {
    return cvSize(d->facesize(), d->facesize());
}

int LibFace::loadConfig(const string& dir) {

    /*
    return d->recognitionCore->loadData(dir);
     */
    return 0;
}

int LibFace::loadConfig(const map<string, string>& config) {
    if(noRecognition()) {
        return 1;
    }
    return d->recognitionCore->loadConfig(config);
}

vector<pair<int, float> > LibFace::recognise(const string& filename, vector<Face*>* faces, int scaleFactor) {

    IplImage* img = cvLoadImage(filename.data(), CV_LOAD_IMAGE_GRAYSCALE); // grayscale
    vector<pair<int, float> > result = this->recognise(img, faces, scaleFactor);
    cvReleaseImage(&img);

    return result;
}

vector<pair<int, float> > LibFace::recognise(const IplImage* img, vector<Face*>* faces, int scaleFactor) {
    vector<pair<int, float> > result;

    if(noRecognition()) {
        return result;
    }

    if (faces->size() == 0) {
        LOG(libfaceWARNING) << " No faces passed to libface::recognise() , not recognizing...";
        return result;
    }

    if (!img) {
        LOG(libfaceWARNING) << " Null image passed to libface::recognise() , not recognizing...";
        return result;
    }

    LOG(libfaceDEBUG) << "Will recognize";


    vector<IplImage*> newFaceImgArr;

    int size = faces->size();

    cout << "Recognise: # of faces: " << size << endl;

    for (int i=0 ; i<size ; i++) {
        Face* face = faces->at(i);
        int x1     = face->getX1();
        int y1     = face->getY1();
        int width  = face->getWidth();
        int height = face->getHeight();

        // Extract face-image from whole-image.
        CvRect rect       = cvRect(x1, y1, width, height);
        IplImage* faceImg = LibFaceUtils::copyRect(img, rect);

        // Make into d->facesize*d->facesize standard-sized image
        IplImage* sizedFaceImg = cvCreateImage(cvSize(d->facesize(), d->facesize()), img->depth, img->nChannels);
        cvResize(faceImg, sizedFaceImg);

        // Extracted. Now push it into the newfaces vector
        newFaceImgArr.push_back(sizedFaceImg);
    }

    // List of Face objects made
    // Now recognize
    for (int i = 0; i < size; ++i)
        result.push_back(d->recognitionCore->recognize(newFaceImgArr.at(i)));

    for (unsigned i=0; i<newFaceImgArr.size(); i++) {
        cvReleaseImage(&newFaceImgArr.at(i));
    }

    return result;
}

vector<pair<int, float> > LibFace::recognise(const char* arr, vector<Face*>* faces, int width, int height, int step, int depth, int channels, int scaleFactor) {
    IplImage* img = LibFaceUtils::charToIplImage(arr, width, height, step, depth, channels);
    return this->recognise(img, faces, scaleFactor);
}

vector<pair<int, float> > LibFace::recognise(vector<Face*>* faces, int scaleFactor) {
    vector<pair<int, float> > result;

    if(noRecognition()) {
        return result;
    }

    if (faces->size() == 0) {
        LOG(libfaceWARNING) << " No faces passed to libface::recognise() , not recognizing.";
        return result;
    }

    LOG(libfaceDEBUG) << "Recognizing.";

    vector<IplImage*> newFaceImgArr;

    int size = faces->size();
    for (int i = 0 ; i < size ; i++) {
        Face* face = faces->at(i);

        LOG(libfaceDEBUG) << "Id is: " << face->getId();

        const IplImage* faceImg = face->getFace();
        IplImage* createdImg    = 0;

        if (!faceImg) {
            LOG(libfaceWARNING) << "Face with null image passed to libface::recognise(), skipping";
            continue;
        }

        if (faceImg->width != d->facesize() || faceImg->height != d->facesize()) {
            // Make into d->facesize*d->facesize standard-sized image
            createdImg = cvCreateImage(cvSize(d->facesize(), d->facesize()), faceImg->depth, faceImg->nChannels);
            cvResize(faceImg, createdImg);
        } else {
            // we need a non-const image for cvEigenDecomposite
            createdImg = cvCloneImage(faceImg);
        }

        result.push_back(d->recognitionCore->recognize(createdImg));

        face->setId(result.at(i).first);

        cvReleaseImage(&createdImg);
    }

    LOG(libfaceDEBUG) << "Size of result = " << result.size();
    return result;
}


void LibFace::training(vector<Face*>* faces, int scaleFactor){

    vector<Mat> images;
    vector<int> labels;

    int size = faces->size();
    for (int i = 0 ; i < size ; i++) {

        Face* face = faces->at(i);

        const IplImage* faceImg = face->getFace();

        images.push_back(cvarrToMat(faceImg));
        labels.push_back(face->getId());
    }

//    Mat tmp = images.at(0);
//    cout << "Row: " << tmp.rows << " Col: " << tmp.cols << endl;
//    cout << "Matrix = " << endl << " " << tmp << endl;

//    for (int i =  0 ; i < labels.size() ; i++)
//        cout << labels[i] << endl;

    d->recognitionCore->training(images,labels);

}

vector<int> LibFace::testing(vector<Face*>* faces){

    vector<int> result;

    vector<Mat> images;
    int size = faces->size();

    for (int i = 0 ; i < size ; i++) {

        Face* face = faces->at(i);

        const IplImage* faceImg = face->getFace();

        int res = d->recognitionCore->testing(cvarrToMat(faceImg));

        result.push_back(res);
    }

    return result;
}

int LibFace::saveConfig(const string& dir) {
    if(noRecognition()) {
        return 1;
    }
    d->recognitionCore->saveConfig(dir);
    return 0;
}

void LibFace::setDetectionAccuracy(double value) {
    if(noDetection()) {
        // cannot return here
    } else {
        d->detectionCore->setAccuracy(value);
    }
}

int LibFace::update(const IplImage* img, vector<Face*>* faces, int scaleFactor) {

    if(noRecognition()) {
        return 1;
    }

    if (faces->size() == 0) {
        LOG(libfaceWARNING) << " No faces passed to update.";
        return 0;
    }

    LOG(libfaceDEBUG) << "Update with faces." << endl;

    vector<Face*>      newFaceArr;

    for (unsigned i=0 ; i<faces->size() ; i++) {
        Face* face = faces->at(i);

        int x1     = face->getX1();
        int y1     = face->getY1();
        int width  = face->getWidth();
        int height = face->getHeight();
        int id     = face->getId();

        LOG(libfaceDEBUG) << "Id is: " << id;

        // Extract face-image from whole-image.
        CvRect rect            = cvRect(x1,y1,width,height);
        IplImage* faceImg      = LibFaceUtils::copyRect(img, rect);

        // Make into standard-sized image
        IplImage* sizedFaceImg = cvCreateImage(cvSize(d->facesize(), d->facesize()), img->depth, img->nChannels);
        cvResize(faceImg, sizedFaceImg);

        face->setFace(sizedFaceImg);
        // Extracted. Now push it into the newfaces vector
        newFaceArr.push_back(face);

    }

    return d->recognitionCore->update(&newFaceArr);

}

int LibFace::update(const char* arr, vector<Face*>* faces, int width, int height, int step, int depth, int channels, int scaleFactor) {
    IplImage* img = LibFaceUtils::charToIplImage(arr, width, height, step, depth, channels);
    return this->update(img, faces, scaleFactor);
}

int LibFace::update(const string& filename, vector<Face*>* faces, int scaleFactor) {
    IplImage* img = cvLoadImage(filename.data(), CV_LOAD_IMAGE_GRAYSCALE); //grayscale
    int result = this->update(img, faces, scaleFactor);
    cvReleaseImage(&img);
    return result;
}

int LibFace::update(vector<Face*> *faces, int scaleFactor) {
    int assignedIDs = 0;

    if(noRecognition()) {
        return assignedIDs;
    }

    if (faces->size() == 0) {
        LOG(libfaceWARNING) << " No faces passed to libface::update() , not updating.";
        return assignedIDs;
    }

    LOG(libfaceDEBUG) << "Update with " << faces->size() << " faces.";

    vector<Face*>      newFaceArr;

    for (unsigned i=0; i<faces->size(); i++) {
        // Copy faces to newFaceArr, dont change the passed face
        Face* face  = faces->at(i);

        LOG(libfaceDEBUG) << "Id is: " << face->getId();

        const IplImage* faceImg = face->getFace();

        if (faceImg->width != d->facesize() || faceImg->height != d->facesize()) {
            // Make into standard-sized image
            IplImage* sizedFaceImg  = cvCreateImage(cvSize(d->facesize() , d->facesize()), faceImg->depth, faceImg->nChannels);
            cvResize(faceImg, sizedFaceImg);
            face->setFace(sizedFaceImg);
        }
        // Extracted. Now push it into the newfaces vector
        newFaceArr.push_back(face);
    }

    assignedIDs = d->recognitionCore->update(&newFaceArr);

    return assignedIDs;
}

bool LibFace::noDetection() const {
    if(d->detectionCore) {
        return 0;
    } else {
        LOG(libfaceERROR) << "Trying to use a function that requires LibFace to be loaded in detection mode, which is not the case.";
        return 1;
    }
}

bool LibFace::noRecognition() const {
    if(d->recognitionCore) {
        return 0;
    } else {
        LOG(libfaceERROR) << "Trying to use a function that requires LibFace to be loaded in recognition mode, which is not the case.";
        return 1;
    }
}

} // namespace libface
