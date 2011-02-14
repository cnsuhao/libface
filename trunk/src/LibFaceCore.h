/** ===========================================================
 * @file
 *
 * This file is a part of libface project
 * <a href="http://libface.sourceforge.net">http://libface.sourceforge.net</a>
 *
 * @date    2010-03-04
 * @brief   Core libface classes.
 * @section DESCRIPTION
 *
 * @author Copyright (C) 2010 by Alex Jironkin
 *         <a href="alexjironkin at gmail dot com">alexjironkin at gmail dot com</a>
 * @author Copyright (C) 2010 by Aditya Bhatt
 *         <a href="adityabhatt at gmail dot com">adityabhatt at gmail dot com</a>
 * @author Copyright (C) 2010 by Gilles Caulier
 *         <a href="mailto:caulier dot gilles at gmail dot com">caulier dot gilles at gmail dot com</a>
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

#ifndef _LIBFACECORE_H_
#define _LIBFACECORE_H_

#if defined (__APPLE__)
#include <cv.h>
#else
#include <opencv/cv.h>
#endif

#include "Face.h"

namespace libface
{

/**
 * Abstract class which all classes for face recognition must implement.
 * It serves as an interface such that they can be easily plugged into backend.
 */
class FACEAPI LibFaceRecognitionCore
{
public:

	virtual ~LibFaceRecognitionCore() {};

	/**
	 * Purely virtual method for retrieving config data from the class. This then can be used by
	 * different applications to save it in the desired way, instead of using default save
	 * method provided here.
	 *
	 * @returns Returns a map of configuration names and their data.
	 */
	virtual std::map<std::string, std::string> getConfig() = 0;

	/**
	 * Purely virtual method for loading config data from a map with keys as string and data
	 * as strings. This needs to be of the same format as returned by getConfig() method.
	 *
	 * @param config A map of configuration names and data associated with it.
	 *
	 * @return 0 if load was successful, non-zero otherwise.
	 */
	virtual int loadConfig(const std::map<std::string, std::string>& config)  = 0;

	/**
	 * Purely virtual method for loading config data from a directory when built in save function was used.
	 *
	 * @param dir A path to the directory with the config file.
	 *
	 * @return 0 if load was successful, non-zero otherwise.
	 */
	virtual int loadConfig(const std::string& dir)  = 0;


	/**
	 * Purely virtual method for saving config data to a directory.
	 *
	 * @param dir A path to the directory where the config will be saved.
	 *
	 * @return 0 if load was successful, non-zero otherwise.
	 */
	virtual int saveConfig(const std::string& dir)  = 0;

	/**
	 * Purely virtual method for updating the system with new array of Face objects.
	 *
	 * @param dataVector A vector of Face objects.
	 *
	 * @return 0 id the update was successful, or positive int above 0 otherwise.
	 */
	virtual int update(std::vector<Face>& dataVector) = 0;

	/**
	 * Purely virtual method for recognising an input image as a face. Returns the ID of the nearest face
	 *
	 * @param test The test IplImage * image to be recognized
	 *
	 * @return The ID of the closest face.
	 */
	virtual std::pair<int, float> recognize(IplImage* test) = 0 ;

	/**
	 * Purely virtual method to return the count of faces in the DB.
	 *
	 * @return The number of faces the DB has been trained with
	 */
	virtual int count() const = 0;
};

// -------------------------------------------------------------------------------------------

class FACEAPI LibFaceDetectCore
{
public:

	virtual ~LibFaceDetectCore() {};

	/**
	 * Purely virtual method for detecting faces in an image.
	 *
	 * @param filename A path to an image file.
	 *
	 * @return Returns a pointer to the vector of Face objects, where each ID is set to -1.
	 */
	virtual std::vector<Face>* detectFaces(const std::string& filename) = 0;

	/**
	 * Purely virtual method for detecting faces in an image.
	 *
	 * @param inputImage A pointer to IplImage for detection.
	 *
	 * @return Returns a pointer to the vector of Face objects, where each ID is set to -1.
	 */
	virtual std::vector<Face>* detectFaces(const IplImage* inputImage) = 0;

	/**
	 * Purely virtual method for getting the accuracy of the detection.
	 *
	 * @return Returns an integer associated with the current accuracy.
	 */
	virtual int accuracy() const = 0;

	/**
	 * Purely virtual method for setting the accuracy of the detection.
	 *
	 * @param value An integer value representing accuracy.
	 */
	virtual void setAccuracy(int value) = 0;
};

} // namespace libface

#endif /* _LIBFACECORE_H_ */
