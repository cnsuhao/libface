/** ===========================================================
 * @file HMMfaces.cpp
 *
 * This file is a part of libface project
 * <a href="http://libface.sourceforge.net">http://libface.sourceforge.net</a>
 *
 * @date    2009-12-27
 * @brief   HMMfaces Main File.
 * @section DESCRIPTION
 *
 * This class is an implementation of HMMfaces algorithm.

 * @author Copyright (C) 2012 by A.H.M. Mahfuzur Rahman
 *         <a href="mamun_nightcrawler at gmail dot com">mamun_nightcrawler at gmail dot com</a>
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


#ifndef HMMfaces_H
#define HMMfaces_H

// LibFace headers
#include "LibFaceConfig.h"
#include "LibFaceCore.h"

// OpenCV headers
#if defined (__APPLE__)
#include <cv.h>
#else
#include <opencv/cv.h>
#endif

// C headers
#include <map>
#include <string>
#include <utility> // for pair
#include <vector>

namespace libface
{

class FACEAPI HMMfaces : public LibFaceRecognitionCore
{
public:

    /**
     * Constructor for HMMfaces. Takes a directory string as argument to determine the location of config xml file.
     *
     * @param dir The directory in which the DB is to be found/created.
     */
    HMMfaces(const std::string& dir = ".");

    /**
     * Copy constructor.
     *
     * @param that Object to be copied.
     */
    HMMfaces(const HMMfaces& that);

    /**
     * Assignment operator.
     *
     * @param that Object to be copied.
     *
     * @return Reference to assignee.
     */
    HMMfaces& operator = (const HMMfaces& that);

    /**
     * Destructor that frees the data variables.
     */
    ~HMMfaces();

    /**
     * Returns the number of unique faces in the database.
     *
     * @return Number of unique faces in the database.
     */
    int count() const;

    /**
     * Get the mapping between config variables and the data. This can be stored and then loaded back into config.
     *
     * @return Returns a config std::map with variable names as keys and data encoded as std::string.
     */
    std::map<std::string, std::string> getConfig();

    /**
     * Method for loading the mapping of config variables and the data back into libface.
     *
     * @param config A std::map config returned by getConfig() method.
     *
     * @return Returns 0 if operation was successful, or positive error codeotherwise.
     */
    int loadConfig(const std::map<std::string, std::string>& config);

    /**
     * Attempts to load config from specified directory.
     *
     * @param dir A directory to look for libface-config.xml file.
     *
     * @return Returns 0 if config was loaded or positive error code otherwise.
     */
    int loadConfig(const std::string& dir);

    /**
     * Method to attempt to compare images with the known projected images. Uses a specified type of
     * distance to see how far away they are from each of the images in the projection.
     *
     * @param input The pointer to IplImage* image, which is to be recognized.
     *
     * @return A pair with ID and closeness of the closest face.
     *
     */
    std::pair<int, float> recognize(IplImage* input);

    /**
     * Saves the config is a given directory.
     *
     * @param dir A std::string path to directory where config should be stored.
     *
     * @return Returns 0 if operation was successful, or positive error code otherwise.
     */
    int saveConfig(const std::string& dir);

    /**
     * Updates the config with a vector of input training faces.
     * If id of the face is -1, then face is added to the end of the faces vector.
     *
     * If id is not -1 and a new id, then face is added to the end of the faces vector.
     *
     * If id is not -1 and it already exist, then given the given faces is projected together with the
     * known face at that position using eigen decomposition and the projected face is stored in it's place.
     *
     * @param newFaceArr The vector of input Face objects
     *
     * @return Returns 0 if update was successful, or positive int otherwise.
     */
    int update(std::vector<Face*>* dataVector);


    /**
     * New Addition
     * Training phase of face recognition
     */
    void training(InputArray src, InputArray labels, int no_principal_components = 0);

    /**
     * New Addition
     * Testing phase of face recognition
     */
    int testing(InputArray src){return 0;}

    int testing(IplImage* img);

    /**
      *
      */
    void updateTest(vector<Face *> *newFaceArr);

private:

    class HMMfacesPriv;
    HMMfacesPriv* const d;
};

} // namespace libface

#endif // HMMfaces_H
