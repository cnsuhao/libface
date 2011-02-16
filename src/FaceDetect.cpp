/** ===========================================================
 * @file
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

#include <iostream>
#include <ctime>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "LibFaceUtils.h"
#include "Haarcascades.h"
#include "FaceDetect.h"
#include "Face.h"

using namespace std;
using namespace cv;

namespace std {
  extern ostream clog;
}

namespace libface
{

class FaceDetect::FaceDetectPriv
{

public:

    FaceDetectPriv()
    {
        cascadeSet = 0;
        storage    = 0;
    }

    Haarcascades* cascadeSet;
    CvMemStorage* storage;
    double        scaleFactor;        // Keeps the scaling factor of the internal image.

    bool          countCertainity;

    int           maximumDistance;    // Maximum distance between two faces to call them unique
    int           minimumDuplicates;  // Minimum number of duplicates required to qualify as a genuine face

    // Tunable values, for accuracy
    float         searchIncrement;
    int           grouping;
    int           minSize[4];
    int           accu;
};

FaceDetect::FaceDetect(const string& cascadeDir)
: d(new FaceDetectPriv)
{
    d->cascadeSet = new Haarcascades(cascadeDir);

    /* Cascades */
    //d->cascadeSet->addCascade("haarcascade_frontalface_alt.xml", 1);   // Weight 1 for frontal default
    d->cascadeSet->addCascade("haarcascade_frontalface_alt2.xml",1);  //default
    //d->cascadeSet->addCascade("haarcascade_profileface.xml", 1);

    d->countCertainity = true;

    this->setAccuracy(1);
}

FaceDetect::~FaceDetect() {
    cvReleaseMemStorage(&d->storage);
    d->cascadeSet->clear();
    delete d->cascadeSet;
    delete d;
}

int FaceDetect::accuracy() const
{
    return d->accu;
}

void FaceDetect::setAccuracy(int i)
{
    if(i >= 1 && i <= 10)
    {
        d->accu = i;
    }
    else
    {
        if (DEBUG)
            cout<<"Bad accuracy value"<<endl;
        return;
    }

    d->maximumDistance   = 20;    // Maximum distance between two faces to call them unique
    d->minimumDuplicates = 1;    // Minimum number of duplicates required to qualify as a genuine face

    // Now adjust values based on accuracy level
    switch(d->accu)
    {
    case 1:
    {
        d->searchIncrement = 1.269F;
        d->minSize[0]      = 1;
        d->minSize[1]      = 20;
        d->minSize[2]      = 26;
        d->minSize[3]      = 35;
        d->grouping        = 1;
        break;
    }

    case 2:
    {
        d->searchIncrement = 1.2F;
        d->minSize[0]      = 1;
        d->minSize[1]      = 20;
        d->minSize[2]      = 30;
        d->minSize[3]      = 40;
        d->grouping        = 3;
        break;
    }
    case 3:
    {
        d->searchIncrement = 1.21F;
        d->minSize[0]      = 1;
        d->minSize[1]      = 20;
        d->minSize[2]      = 26;
        d->minSize[3]      = 35;
        d->grouping        = 3;
        break;
    }
    case 4:
    {
        d->searchIncrement = 1.268F;
        d->minSize[0]      = 1;
        d->minSize[1]      = 30;
        d->minSize[2]      = 40;
        d->minSize[3]      = 50;
        d->grouping        = 2;
        break;
    }
    default:
    {
        // TODO: missing case for compiler.
        break;
    }
    };
}

vector<Face>* FaceDetect::cascadeResult(const IplImage* inputImage, CvHaarClassifierCascade* casc, CvSize faceSize)
{
    // Clear the memory d->storage which was used before
    cvClearMemStorage(d->storage);

    vector<Face>* result = new vector<Face>();

    CvSeq* faces = 0;

    // Create two points to represent the face locations
    CvPoint pt1, pt2;

    // Check whether the cascade has loaded successfully. Else report and error and quit
    if (!casc)
    {
        cerr << "ERROR: Could not load classifier cascade." << endl;
        return result;
    }

    // Find whether the cascade is loaded, to find the faces. If yes, then:
    if (casc)
    {
        //TODO: Also may give a weight to the cascades, maybe alt-1, default and alt2 - 0.8?

        // There can be more than one face in an image. So create a growable sequence of faces.
        // Detect the objects and store them in the sequence
        clock_t detect;

        if (DEBUG)
            detect = clock();

        faces = cvHaarDetectObjects(inputImage,
                casc,
                d->storage,
                d->searchIncrement,             // Increase search scale by 5% everytime
                d->grouping,                              // Drop groups of less than 2 detections
                CV_HAAR_DO_CANNY_PRUNING,
                faceSize                        // Minimum face size to look for
        );

        if (DEBUG)
        {
            detect = clock() - detect;
            printf("Detection took: %f secs.\n", (double)detect / ((double)CLOCKS_PER_SEC));
        }

        // Loop the number of faces found.
        for (int i = 0; i < (faces ? faces->total : 0); i++)
        {
            // Create a new rectangle for drawing the face

            CvRect* roi = (CvRect*) cvGetSeqElem(faces, i);

            // Find the dimensions of the face,and scale it if necessary.
            float boxShrink = 0.1;

            pt1.x     = (int)(roi->x  * d->scaleFactor * 1);
            pt2.x     = (int)((roi->x + roi->width)  * d->scaleFactor);
            pt1.y     = (int)(roi->y  * d->scaleFactor * 1);
            pt2.y     = (int)((roi->y + roi->height) * d->scaleFactor);

            //Make box a bit tighter
            int width = pt2.x - pt1.x;
            int height = pt2.y - pt1.y;

            pt1.x = pt1.x + (int)(width*boxShrink);
            pt1.y = pt1.y + (int)(height*boxShrink);
            pt2.x = pt2.x - (int)(width*boxShrink);
            pt2.y = pt2.y - (int)(height*boxShrink);

            Face face = Face(pt1.x,pt1.y,pt2.x,pt2.y);

            result->push_back(face);
        }

        //Please don't delete next line even if commented out. It helps with testing intermediate results.
        //LibFaceUtils::showImage(inputImage, result);

    }

    return result;
}

vector<Face> FaceDetect::finalFaces(const IplImage* inputImage, vector<vector<Face> > combo, int maxdist, int mindups)
{
    clock_t      finalStage;
    vector<Face> finalResult;
    vector<int>  genuineness;

    if (DEBUG)
        finalStage = clock();

    // Make one long vector of all faces

    for (unsigned int i = 0; i < combo.size(); ++i)
    {
        for (unsigned int j = 0; j < combo[i].size(); ++j)
            finalResult.push_back(combo[i].at(j));
    }

    if (DEBUG)
    {
        finalStage = clock() - finalStage;
    }

    /*
    Now, starting from the left, take a face and compare with rest. If distance is less than a threshold, 
    consider them to be "overlapping" face frames and delete the "duplicate" from the vector.
    Remember that only faces to the RIGHT of the reference face will be deleted.
     */
    if (DEBUG)
        finalStage = clock();

    int ctr = 0;
    for (unsigned int i = 0; i < finalResult.size(); ++i)
    {
        int duplicates = 0;
        for (unsigned int j = i + 1; j < finalResult.size(); ++j)    // Compare with the faces to the right
        {
            ctr++;
            if (LibFaceUtils::distance(finalResult[i], finalResult[j]) < maxdist)
            {
                finalResult.erase(finalResult.begin() + j);
                duplicates++;
                j--;

            }
        }
        genuineness.push_back(duplicates);
        if (duplicates < mindups)    // Less duplicates, probably not genuine, kick it out
        {
            genuineness.erase(genuineness.begin() + i);
            finalResult.erase(finalResult.begin() + i);
            i--;
        }
        /* 
        Note that the index of the reference element will be the same as the index of it's number of duplicates
        in the genuineness vector, so win-win!.
         */
    }

    if (DEBUG)
    {
        printf("Faces parsed : %d, number of final faces : %d\n", ctr, (int)genuineness.size());
        finalStage = clock() - finalStage;
        printf("Prunning took: %f sec.\n", (double)finalStage / ((double)CLOCKS_PER_SEC));
    }

    if (finalResult.size() == 0)
    {
        return finalResult;
    }
    vector<Face> returnFaces;

    for (unsigned int j = 0; j < finalResult.size(); ++j)
    {
        Face face         = finalResult[j];

        //Extract face-image from whole-image.
        CvRect rect       = cvRect(face.getX1(),face.getY1(),face.getWidth(),face.getHeight());
        IplImage* faceImg = LibFaceUtils::copyRect(inputImage, rect);
        face.setFace(faceImg);

        returnFaces.push_back(face);
    }

    return returnFaces;
}

int FaceDetect::getRecommendedImageSizeForDetection()
{
    return 800; // area, with typical photos, about 500000
}

std::vector<Face>* FaceDetect::detectFaces(const IplImage* inputImage)
{
    if(inputImage->width < 50 || inputImage->height < 50 || inputImage->imageData == 0)
    {
        cout<<"Bad image given, not detecting faces."<<endl;
        return new vector<Face>();
    }

    IplImage* imgCopy = cvCloneImage(inputImage);
    clock_t init, final;

    init           = clock();

    int faceSize   = d->minSize[0];
    IplImage* temp = 0;
    // BIG problems with small images if this isn't initialized here, it remains 0
    // Not an issue with big images because d->scaleFactor is passed by reference to the resizer and automatically set.
    d->scaleFactor = 1;

    int inputArea  = inputImage->width*inputImage->height;

    if (DEBUG)
    	printf("Input area : %d\n", inputArea);
        //clog << "Input area: " << inputArea << endl;


    if (inputArea > 7000000)
    {
        temp = libface::LibFaceUtils::resizeToArea(inputImage, 786432, d->scaleFactor);

        if (DEBUG)
            printf("Image scaled to %d pixels\n", 786432);

        this->setAccuracy(3);
    }
    else if (inputArea > 5000000)
    {
        temp = libface::LibFaceUtils::resizeToArea(inputImage, 786432, d->scaleFactor);

        if (DEBUG)
            printf("Image scaled to %d pixels\n",786432);

        this->setAccuracy(2);
    }
    else if (inputArea > 2000000)
    {
        temp = libface::LibFaceUtils::resizeToArea(inputImage, 786432, d->scaleFactor);

        if (DEBUG)
            printf("Image scaled to %d pixels\n", 786432);

        float ratio = (float) inputImage->width/inputImage->height;
        if ( ratio == (float) 4/3 )
            this->setAccuracy(4);
        else
            this->setAccuracy(4);


    }


    // This is the combination of all the resulting faces from each cascade in the set 1228800
    // Now loop through each cascade, apply it, and get back a vector of detected faces
    //vector< vector<Face> > resultCombo;
    vector<Face>* faces;

    d->storage = cvCreateMemStorage(0);
    for (int i = 0; i < d->cascadeSet->getSize(); ++i)
    {
        IplImage* constTemp = temp ? temp : cvCloneImage(inputImage);
        faces                = this->cascadeResult(constTemp, d->cascadeSet->getCascade(i).haarcasc, cvSize(faceSize,faceSize));
    }
    cvReleaseMemStorage(&d->storage);

    final = clock()-init;
    if (DEBUG)
        cout<<"Total time taken : " << (double)final / ((double)CLOCKS_PER_SEC)<< "seconds" << endl;

    // After intelligently "merging" overlaps of face regions by different cascades,
    // this returns the final list of faces. Allow a max distance of 15.
    //vector<Face> ret = finalFaces(inputImage, resultCombo, d->maximumDistance, 0);

    for(int i=0; i<faces->size();i++) {

        CvRect roi = cvRect(faces->at(i).getX1(), faces->at(i).getY1(), faces->at(i).getWidth(), faces->at(i).getHeight());
        cvSetImageROI(imgCopy, roi);

        IplImage *roiImg = cvCreateImage( cvSize(roi.width, roi.height), inputImage->depth, inputImage->nChannels );
        cvCopy(imgCopy, roiImg);
        cvResetImageROI(imgCopy);

        faces->at(i).setFace(roiImg);
    }

    cvReleaseImage(&imgCopy);

    if (temp)
        cvReleaseImage(&temp);
    return faces;
}

vector<Face>* FaceDetect::detectFaces(const string& filename)
{
    // Create a new image based on the input image
    IplImage* img = cvLoadImage(filename.data(), CV_LOAD_IMAGE_GRAYSCALE);

    vector<Face>* faces = detectFaces(img);

    cvReleaseImage(&img);

    return faces;
}

} // namespace libface
