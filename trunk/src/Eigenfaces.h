/** ===========================================================
 * @file
 *
 * This file is a part of libface project
 * <a href="http://libface.sourceforge.net">http://libface.sourceforge.net</a>
 *
 * @date    2009-12-27
 * @brief   Eigenfaces parser
 * @section DESCRIPTION
 *
 * This class is somewhat based on an implementation of Eigenfaces in a tutorial, originally
 * done by Robin Hewitt.
 *
 * @author Copyright (C) 2009-2010 by Alex Jironkin
 *         <a href="alexjironkin at gmail dot com">alexjironkin at gmail dot com</a>
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

#ifndef _EIGENFACES_H_
#define _EIGENFACES_H_

#include <string>

#include "LibFaceConfig.h"
#include "LibFaceCore.h"

namespace libface
{

class FACEAPI Eigenfaces : public LibFaceRecognitionCore
{
public:

    /**
     * Constructor for Eigenfaces. Takes a directory string as argument to know the DB location
     * @param dir The directory in which the DB is to be found/created
     */
    Eigenfaces(const std::string& dir = ".");

    /**
     * Destructor
     */
    ~Eigenfaces();

    /**
     * Get the mapping between config variables and the data. This can be stored and then
     * loaded back into config.
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
     * Saves the config is a given directory.
     *
     * @param dir A std::string path to directory where config should be stored.
     *
     * @return Returns 0 if operation was successful, or positive error code otherwise.
     */
    int saveConfig(const std::string& dir);

    /**
     * Updates the DB set with a vector of input training faces.
     * @param newFaceArr The vector of input Face objects
     */
    std::vector<int> update(std::vector<Face>&);

    /**
     * Method to attempt to compare images with the known projected images. Uses a specified type of
     * distance to see how far away they are from each of the images in the projection.
     * @param DISTANCE_TYPE Type of distance - EUCLIDEAN, MAHALANOBIS
     * @param input The vector of IplImage * images, which are the faces to be recognized
     * @return A pair with ID and closeness of the closest face
     * TODO: Implement the usage of the already available const int CUT_OFF to reject faces that are too far
     */
    std::pair<int, double> recognize(IplImage*);

    /**
     * Returns the number of all faces trained in the database.
     */
    int count() const;

    /**
     * Returns the number of faces of a person with the specified id in the database.
     */
    int count(int id) const;

private:

    class EigenfacesPriv;
    EigenfacesPriv* const d;
};

} // namespace libface

#endif /* _EIGENFACES_H_ */
