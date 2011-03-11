/** ===========================================================
 * @file
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

#include <sys/stat.h>

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cerrno>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iterator>

#include "LibFace.h"
#include "LibFaceUtils.h"
#include "FaceDetect.h"
#include "Face.h"
#include "Log.h"

using namespace std;


extern std::ostream std::clog;


namespace libface {

class LibFace::LibFacePriv {

public:

	LibFacePriv() {
		detectionCore   = 0;
		recognitionCore = 0;
		lastImage       = 0;
	}

	Mode                    type;
	std::string             cascadeDir;
	LibFaceDetectCore*      detectionCore;
	LibFaceRecognitionCore* recognitionCore;
	IplImage*               lastImage;
	std::string             lastFileName;

	static int              facesize() { return 120; }
};

LibFace::LibFace(Mode type, const string& configDir, const string& cascadeDir)
: d(new LibFacePriv) {

	d->type = type;

	LOG(libfaceINFO) << "Cascade directory located in : " << cascadeDir;

	// We don't need face recognition if we just want detection, and vice versa.
	// So there is a case for everything.
	switch (d->type) {
	case DETECT:
		d->cascadeDir      = cascadeDir;
		d->detectionCore   = new FaceDetect(d->cascadeDir);
		break;
	case EIGEN:
		d->recognitionCore = new Eigenfaces(configDir);
		break;
	case HMM:
		d->recognitionCore = new HMMfaces();
		break;
	default:    // Initialize both detector and Eigenfaces
	d->cascadeDir      = cascadeDir;
	d->detectionCore   = new FaceDetect(d->cascadeDir);
	d->recognitionCore = new Eigenfaces(configDir);
	//d->recognitionCore = new HMMfaces();
	break;
	}
}

LibFace::~LibFace() {
	switch(d->type) {
	case DETECT:
		delete d->detectionCore;
		break;
	case EIGEN:
		delete d->recognitionCore;
		break;
	default:
		delete d->detectionCore;
		delete d->recognitionCore;
		break;
	}
	cvReleaseImage(&d->lastImage);

	delete d;
}

int LibFace::count() const {
	return d->recognitionCore->count();
}

vector<Face*>* LibFace::detectFaces(const string& filename, int scaleFactor) {
	if(filename.length() == 0) {
		LOG(libfaceWARNING) << "No image passed for detection";
		return 0;
	}
	//Check if image was already loaded once, by checking last loaded filename.
	if (filename != d->lastFileName) {
		d->lastFileName = filename;
		cvReleaseImage(&d->lastImage);
		d->lastImage    = cvLoadImage(filename.data(), CV_LOAD_IMAGE_GRAYSCALE);
	}

	return d->detectionCore->detectFaces(d->lastImage);
}

vector<Face*>* LibFace::detectFaces(const char* arr, int width, int height, int step, int depth, int channels, int scaleFactor) {
	IplImage* image = LibFaceUtils::charToIplImage(arr, width, height, step, depth, channels);
	return d->detectionCore->detectFaces(image);
}

vector<Face*>* LibFace::detectFaces(const IplImage* image) {
	return d->detectionCore->detectFaces(image);
}

map<string,string> LibFace::getConfig() {
	map<string,string> result = d->recognitionCore->getConfig();
	return result;
}

double LibFace::getDetectionAccuracy() const {
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

	for (int i=0; i<newFaceImgArr.size(); i++)
		cvReleaseImage(&newFaceImgArr[i]);

	return result;
}

vector<pair<int, float> > LibFace::recognise(const char* arr, vector<Face*>* faces, int width, int height, int step, int depth, int channels, int scaleFactor) {
	IplImage* img = LibFaceUtils::charToIplImage(arr, width, height, step, depth, channels);
	return this->recognise(img, faces, scaleFactor);
}

vector<pair<int, float> > LibFace::recognise(vector<Face*>* faces, int scaleFactor) {
	vector<pair<int, float> > result;

	if (faces->size() == 0) {
		LOG(libfaceWARNING) << " No faces passed to libface::recognise() , not recognizing.";
		return result;
	}

	LOG(libfaceDEBUG) << "Recognizing.";

	vector<IplImage*> newFaceImgArr;

	int size = faces->size();
	for (int i = 0 ; i < size ; i++) {
		Face* face = faces->at(i);
		int id     = face->getId();

		LOG(libfaceDEBUG) << "Id is: " << id;

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

int LibFace::saveConfig(const string& dir) {
	int result  = 0;
	d->recognitionCore->saveConfig(dir);
	return result;
}

void LibFace::setDetectionAccuracy(double value) {
	d->detectionCore->setAccuracy(value);
}

int LibFace::update(const IplImage* img, vector<Face*>* faces, int scaleFactor) {
	int assignedIDs;


	if (faces->size() == 0) {
		LOG(libfaceWARNING) << " No faces passed to update.";
		return assignedIDs;
	}

	LOG(libfaceDEBUG) << "Update with faces." << endl;

	vector<Face*>      newFaceArr;
	vector<IplImage*> createdImages;

	int size = faces->size();
	for (int i=0 ; i<size ; i++) {
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
		createdImages.push_back(sizedFaceImg);

		face->setFace(sizedFaceImg);
		// Extracted. Now push it into the newfaces vector
		newFaceArr.push_back(face);

	}
	assignedIDs = d->recognitionCore->update(&newFaceArr);

	for (int i=0; i<createdImages.size(); i++)
		cvReleaseImage(&createdImages[i]);

	return assignedIDs;
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

	if (faces->size() == 0) {
		LOG(libfaceWARNING) << " No faces passed to libface::update() , not updating.";
		return assignedIDs;
	}

	LOG(libfaceDEBUG) << "Update with faces.";

	vector<Face*>      newFaceArr;
	vector<IplImage*> createdImages;

	int size = faces->size();
	for (int i=0; i<size; i++) {
		// Copy, dont change the passed face
		Face* face  = faces->at(i);

		LOG(libfaceDEBUG) << "Id is: " << face->getId();


		const IplImage* faceImg = face->getFace();
		if (faceImg->width != d->facesize() || faceImg->height != d->facesize()) {
			// Make into standard-sized image
			IplImage* sizedFaceImg  = cvCreateImage(cvSize(d->facesize() , d->facesize()), faceImg->depth, faceImg->nChannels);
			cvResize(faceImg, sizedFaceImg);
			face->setFace(sizedFaceImg);
			createdImages.push_back(sizedFaceImg);
		}
		// Extracted. Now push it into the newfaces vector
		newFaceArr.push_back(face);
	}

	assignedIDs = d->recognitionCore->update(&newFaceArr);

	for (int i=0; i<createdImages.size(); i++)
		cvReleaseImage(&createdImages[i]);

	return assignedIDs;
}

} // namespace libface
