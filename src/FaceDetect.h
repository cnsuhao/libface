/** ===========================================================
 * @file FaceDetect.h
 *
 * This file is a part of libface project
 * <a href="http://libface.sourceforge.net">http://libface.sourceforge.net</a>
 *
 * @date    2010-01-03
 * @brief   Class to perform faces detection.
 * @section DESCRIPTION
 *
 * @author Copyright (C) 2010 by Alex Jironkin
 *         <a href="alexjironkin at gmail dot com">alexjironkin at gmail dot com</a>
 * @author Copyright (C) 2010 by Aditya Bhatt
 *         <a href="adityabhatt at gmail dot com">adityabhatt at gmail dot com</a>
 * @author Copyright (C) 2010 by Gilles Caulier
 *         <a href="mailto:caulier dot gilles at gmail dot com">caulier dot gilles at gmail dot com</a>
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

#ifndef _FACEDETECT_H_
#define _FACEDETECT_H_

// LibFace headers
#include "LibFaceCore.h"

// OpenCV headers
#if defined (__APPLE__)
#include <cv.h>
#else
#include <opencv/cv.h>
#endif

// C headers
#include <string>
#include <vector>

namespace libface
{

// forward declaration
class Face;

class FACEAPI FaceDetect : public LibFaceDetectCore
{
public:

    /**
     * Default constructor for the FaceDetect class. Initialises the cascade to a cascade in the
     * specified directory.
     *
     * @param cascadeDir A path to the directory where classifier is.
     *
     */
    FaceDetect(const std::string& cascadeDir);

    /**
     * Copy constructor.
     *
     * @param that Object to be copied.
     */
    FaceDetect(const FaceDetect& that);

    /**
     * Assignment operator.
     *
     * @param that Object to be copied.
     *
     * @return Reference to assignee.
     */
    FaceDetect& operator = (const FaceDetect& that);

    /**
     *  Default destructor. Deletes d.
     */
    ~FaceDetect();

    /**
     * Detects faces in an input image.
     *
     * @param inputImage A pointer to the image in which faces are to be detected.
     *
     * @return The vector of detected faces.
     */
    std::vector<Face*>* detectFaces(const IplImage* inputImage);

    /**
     * Inherited method from LibFaceDetectCore. A slightly different interface where you can specify
     * full path to image.
     *
     * @param filename A full path to the image.
     *
     * @return Returns a vector of Face objects. Each object hold information about 1 face.
     */
    std::vector<Face*>* detectFaces(const std::string& filename);

    /**
     * Get accuracy of face detection on a five-point scale. The default is 4.
     *
     * @return Accuracy.
     */
    int accuracy() const;

    /**
     * Set the accuracy of face detection on a five-point scale.
     *
     * @param value Desired accuracy.
     */
    void setAccuracy(int value);

    /**
     * Returns the image size (one dimension) recommended for face detection. If the image is considerably larger, it will be rescaled automatically.
     *
     * @return Recommended image size for face detection.
     */
    static int getRecommendedImageSizeForDetection();

private:

    /**
     *  Inherited method from LibFaceDetectCore for detecting faces in an image using a single cascade. Uses CANNY_PRUNING at present.
     *
     *  @param inputImage A pointer to the IplImage representing image of interest.
     *  @param casc The CvClassClassifierCascade pointer to be used for the detection.
     *  @param faceSize A cvSize that specifies the minimum size of faces to be detected.
     *
     *  @return Returns a vector of Face objects. Each object hold information about 1 face.
     */
    std::vector<Face*>* cascadeResult(const IplImage* inputImage, CvHaarClassifierCascade* casc, CvSize faceSize = cvSize(10, 10));

    /**
     * Returns the final faces from the detection results of multiple cascades.
     *
     * @param combo A vector of a vector of faces, each component vector is the detection result of a single cascade.
     * @param maxdist The maximum allowable distance between two duplicates, if two faces are further apart than this, they are not duplicates.
     * @param mindups The minimum number of duplicate detections required for a face to qualify as genuine.
     *
     * @return The vector of the final faces.
     */
    std::vector<Face> finalFaces(const IplImage*, std::vector< std::vector<Face> >, int maxdist, int mindups);

    class FaceDetectPriv;
    FaceDetectPriv* const d;
};

} // namespace libface

#endif /* _FACEDETECT_H_ */
