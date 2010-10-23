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

#ifdef WIN32
// To avoid warnings from MSVC compiler about OpenCV headers
#pragma warning( disable : 4996 )
#endif // WIN32

#include <sys/stat.h>

#include <ctime>
#include <cctype>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iterator>

#include <opencv/cvaux.h>
#include <opencv/cv.h>

//TODO: Check where the data needs to be released to reduce memoryconsumption.
#include "Eigenfaces.h"
#include "FaceDetect.h"
#include "LibFaceUtils.h"

using namespace std;

namespace libface
{

class Eigenfaces::EigenfacesPriv
{

public:

    EigenfacesPriv()
    {
        CUT_OFF               = 10000000.0; //50000000.0;
        UPPER_DIST            = 10000000;
        LOWER_DIST            = 10000000;

        nTrainFaces           = 0;
        nEigens               = 0;

        //personNumTruthMat   = 0;          // Array of person numbers
        pAvgTrainImg          = 0;
        eigenValMat           = 0;
        projectedTrainFaceMat = 0;
    }

    /**
     * Performs PCA on the current training data, projects the training faces, and stores them in a DB.
     */
    void learn();

    /**
     * Performs PCA on the current training data
     */
    void doPCA();

    /**
     * Finds the nearest neighbor of a projected face of float* type, using the specified distance type
     * @param projectedTestFace The projected test face whose neighbor is needed
     * @param distance_type Euclidean = 0, Mahalanobis = 1
     * @return A pair of the index (NOT ID) as int, and the least distance, as double (measure of certainity)
     */
    std::pair<int, double> findNearestNeighbor(float* projectedTestFace) const;

    /**
     * Converts integer to string, convenience function. TODO: Move to Utils
     * @param x The integer to be converted to std::string
     * @return Stringified version of integeer
     */
    inline string stringify(unsigned int x) const;
    void clearImages(std::vector<IplImage*>& list);
    void clearTrainingStructures();

public:

    int                    nTrainFaces;                // Number of training images
    int                    nEigens;                    // Number of Eigenvalues

    // Face data members, stored in the DB
    std::vector<IplImage*> faceImgArr;                 // Array of face images
    std::vector<int>       idArr;

    // Config data members
    std::string            dbFile;
    int                    dbNum;
    int                    totalTrainedFaces;
    std::string            configFile;

    // To be used while calculating stuff
    IplImage*              pAvgTrainImg;               // The Average Image
    std::vector<IplImage*> eigenVectArr;               // The array of eigenvectors
    CvMat*                 eigenValMat;                // eigenvalues
    CvMat*                 projectedTrainFaceMat;      // Projected training faces

    std::map<int, int>     indexIdMap;
    std::map<int, int>     idCountMap;

    double                 CUT_OFF;
    double                 UPPER_DIST;
    double                 LOWER_DIST;
};

void Eigenfaces::EigenfacesPriv::clearImages(std::vector<IplImage*> &list)
{
    for (std::vector<IplImage*>::iterator it = list.begin(); it != list.end(); ++it)
        cvReleaseImage(&(*it));
    list.clear();
}

void Eigenfaces::EigenfacesPriv::clearTrainingStructures()
{
    cvReleaseImage(&pAvgTrainImg);             // The Average Image
    clearImages(eigenVectArr);                 // The array of eigenvectors
    cvReleaseMat(&eigenValMat);                // eigenvalues
    cvReleaseMat(&projectedTrainFaceMat);      // Projected training faces
}

/**
 * Performs PCA on the current training data, projects the training faces, and stores them in a DB.
 */
void Eigenfaces::EigenfacesPriv::learn()
{
    int i;

    // Load the training data. This will store the image data in faceImgArr and return the number of faces in nTrainFaces.
    // The person ID numebers will be stored in personNumTruthMat.
    nTrainFaces = faceImgArr.size();

    if(nTrainFaces == 1)
    {
        IplImage* junkImage = cvCreateImage(cvSize(faceImgArr[0]->width, faceImgArr[0]->height),
                                            faceImgArr[0]->depth, 
                                            faceImgArr[0]->nChannels);

        // draw a line across it - it is claimed that cvCreateImage creates a blank image, we don't want that
        cvLine(junkImage, cvPoint(1, 1), cvPoint(15, 15), cvScalar(255) );

        IplImage* face = faceImgArr.at(0);
        faceImgArr.clear();

        faceImgArr.push_back(junkImage);
        indexIdMap[0]  = -1;
        idCountMap[-1] = 1;
        faceImgArr.push_back(face);
        indexIdMap[1]  = 0;
        idCountMap[0]  = 1;

        // Now, the faces are ordered as ID #0 => junk face, ID #1 => first real face, ID #3 => second real face, so on
        nTrainFaces    += 1;
    }

/*
    if(DEBUG)
    {
        cout << "Performing PCA..." << endl;
    }
*/

    clearTrainingStructures();

    // Do PCA
    doPCA();
/*
    if (DEBUG)
    {
        cout << "PCA complete." << endl;
    }
*/

    // Create a matrix to store the projected faces
    projectedTrainFaceMat = cvCreateMat(nTrainFaces, nEigens, CV_32FC1);

/*
    if (DEBUG)
    {
        cout << "Projecting the training images..." << endl;
    }
*/

    // Project the training images onto the PCA subspace
    for (i = 0; i < nTrainFaces; ++i)
    {
        // Perform the projection
        cvEigenDecomposite( faceImgArr[i],
                            nEigens,
                            &eigenVectArr[0],
                            0,
                            0,
                            pAvgTrainImg,
                            projectedTrainFaceMat->data.fl + i*nEigens );
    }

/*
    if (DEBUG)
    {
        cout << "Projection complete." << endl;
        LibFaceUtils::printMatrix(projectedTrainFaceMat);
    }
*/
}

void Eigenfaces::EigenfacesPriv::doPCA()
{
    CvTermCriteria calcLimit;
    CvSize         faceImgSize;

    // Set the number of eigenvalues to use
    nEigens            = nTrainFaces - 1;

    // Allocate the eigenvector images
    faceImgSize.width  = faceImgArr[0]->width;
    faceImgSize.height = faceImgArr[0]->height;

    // Memory is cleared before calling this method

    // Allocate memory for images
    eigenVectArr = std::vector<IplImage*>(nEigens);

    for (int i = 0; i < nEigens; ++i)
    {
        eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);
    }

    // Allocate the eigenvalue array
    eigenValMat  = cvCreateMat(1, nEigens, CV_32FC1);

    // Allocate the average image
    pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

    // Set PCA's termination criterion using cvTermCriteria - tell it to compute each eigenvalue and then stop
    calcLimit    = cvTermCriteria(CV_TERMCRIT_ITER, nEigens, 1);

/*
    if (DEBUG)
    {
        cout << "Calculating the Average Image, Eigenvalues, and Eigenvectors..." << endl;
    }
*/
    // Invoke cvCalcEigenObjects to compute average image, eigenvalues, and eigenvectors
    cvCalcEigenObjects( nTrainFaces,
                        (void*)&faceImgArr[0],
                        (void*)&eigenVectArr[0],
                        CV_EIGOBJ_NO_CALLBACK,
                        0,
                        NULL,
                        &calcLimit,
                        pAvgTrainImg,
                        eigenValMat->data.fl );

/*
     if (DEBUG)
     {
         cout << "Calculation complete." << endl;
     }
*/
}

pair<int, double> Eigenfaces::EigenfacesPriv::findNearestNeighbor(float* projectedTestFace) const
{
    double least_distance;
    int    iNearest = -1;

    double leastDistSq = DBL_MAX;
    int    i, iTrain;

    for (iTrain = 0; iTrain < nTrainFaces; ++iTrain)
    {
        double distSq = 0;
        for (i = 0; i < nEigens; ++i)
        {
            float d_i = projectedTestFace[i] - projectedTrainFaceMat->data.fl[iTrain * nEigens + i];
            distSq    += d_i*d_i;
        }

        if (distSq < leastDistSq)
        {
            leastDistSq = distSq;
            iNearest    = iTrain;
        }
    }

    least_distance = leastDistSq;

    return make_pair<int, double>(iNearest, least_distance);
}

string Eigenfaces::EigenfacesPriv::stringify(unsigned int x) const
{
    ostringstream o;
    if (!(o << x))
    {
        if (DEBUG)
        {
            cerr << "Could not convert" << endl;
        }
    }
    return o.str();
}

Eigenfaces::Eigenfaces(const string& dir)
          : d(new EigenfacesPriv)
{
    d->indexIdMap.clear();
    d->idCountMap.clear();

    struct stat stFileInfo;
    d->configFile = dir + string("/libface-config.xml");

    if(DEBUG)
        cout << "Config location: " << d->configFile << endl;

    int intStat = stat(d->configFile.c_str(),&stFileInfo);
    if (intStat == 0)
    {
        if (DEBUG)
        {
            cout << "libface config file exists." << endl;
        }
        loadConfig(dir);
    }
    else
    {
        if (DEBUG)
        {
            cout << "libface config file does not exist." << endl;
        }
    }
}

Eigenfaces::~Eigenfaces()
{
    d->clearImages(d->faceImgArr);
    d->clearTrainingStructures();

    for (std::vector<IplImage*>::iterator it = d->faceImgArr.begin(); it != d->faceImgArr.end(); ++it)
        cvReleaseImage(&(*it));

    delete d;
}

map<string, string> Eigenfaces::getConfig()
{
    map<string, string> config;
    int nIds = d->idCountMap.size();

    char nEigensStr[5];
    sprintf(nEigensStr,"%d", d->nEigens);
    config["nEigens"]               = string(nEigensStr);

    char nTrainStr[6];
    sprintf(nTrainStr,"%d", d->nTrainFaces);
    config["nTrainfaces"]           = string(nTrainStr);

    char idCountStr[6];
    sprintf(idCountStr,"%d", nIds);
    config["nIds"]                  = string(idCountStr);

    config["projectedTrainFaceMat"] = LibFaceUtils::matrixToString(d->projectedTrainFaceMat);
    config["eigenValMat"]           = LibFaceUtils::matrixToString(d->eigenValMat);
    config["avgTrainImg"]           = LibFaceUtils::imageToString(d->pAvgTrainImg);

    for ( int i = 0; i < d->nTrainFaces; ++i )
    {
        char facename[200];
        sprintf(facename, "person_%d", i);
        config[string(facename)] = LibFaceUtils::imageToString(d->faceImgArr[i]);
    }

    for ( int j = 0; j < d->nEigens; ++j )
    {
        char varname[200];
        sprintf(varname, "eigenVect_%d", j);
        config[string(varname)] = LibFaceUtils::imageToString(d->eigenVectArr[j]);
    }

    for ( int k = 0; k < nIds; ++k )
    {
        char idname[200];
        sprintf(idname, "indexIdMap_%d", k);
        char data[5];
        sprintf(data,"%d", d->indexIdMap[k]);
        config[string(idname)] = string(data);
    }

    for ( int k = 0; k < nIds; ++k )
    {
        char idname[200];
        int  id = d->indexIdMap[k];    // Retrieve Id from the indexIdMap
        char data[5];
        sprintf(data,"%d", d->idCountMap[id]);
        sprintf(idname, "idCountMap_%d", id);
        config[string(idname)] = string(data);
    }

    return config;
}

int Eigenfaces::loadConfig(const string& dir)
{
    d->configFile = dir + string("/libface-config.xml");

    if (DEBUG)
    {
        cout << "Load training data" << endl;
    }

    CvFileStorage* fileStorage = cvOpenFileStorage(d->configFile.data(), 0, CV_STORAGE_READ);

    if (!fileStorage)
    {
        if (DEBUG)
        {
            cout << "Can't open config file for reading :" << d->configFile << endl;
        }
        return 1;
    }

    d->clearTrainingStructures();

    d->nEigens               = cvReadIntByName(fileStorage,         0, "nEigens",               0);
    d->nTrainFaces           = cvReadIntByName(fileStorage,         0, "nTrainFaces",           0);
    int nIds                 = cvReadIntByName(fileStorage,         0, "nIds",                  0);
    d->eigenValMat           = (CvMat*)cvReadByName(fileStorage,    0, "eigenValMat",           0);
    d->projectedTrainFaceMat = (CvMat*)cvReadByName(fileStorage,    0, "projectedTrainFaceMat", 0);
    d->pAvgTrainImg          = (IplImage*)cvReadByName(fileStorage, 0, "avgTrainImg",           0);
    d->clearImages(d->eigenVectArr);
    d->eigenVectArr          = std::vector<IplImage*>(d->nTrainFaces);

    //LibFaceUtils::printMatrix(d->projectedTrainFaceMat);

    for ( int i = 0; i < d->nTrainFaces; ++i )
    {
        char facename[200];
        sprintf(facename, "person_%d", i);
        d->faceImgArr.push_back( (IplImage*)cvReadByName(fileStorage, 0, facename, 0) );
    }

    for ( int j = 0; j < d->nEigens; ++j )
    {
        char varname[200];
        sprintf(varname, "eigenVect_%d", j);
        d->eigenVectArr[j] = (IplImage*)cvReadByName(fileStorage, 0, varname, 0);
    }

    for ( int k = 0; k < nIds; ++k )
    {
        char idname[200];
        sprintf(idname, "indexIdMap_%d", k);
        d->indexIdMap[k] = cvReadIntByName(fileStorage, 0, idname, 0);
    }

    for ( int k = 0; k < nIds; ++k)
    {
        char idname[200];
        int id            = d->indexIdMap[k];    // Retrieve Id from the indexIdMap
        sprintf(idname, "idCountMap_%d", id);
        d->idCountMap[id] = cvReadIntByName(fileStorage, 0, idname, 0);

        //cvWrite(fileStorage, idname, (int*)&d->idCountMap[k], cvAttrList(0,0));
    }

    // Release file storage
    cvReleaseFileStorage(&fileStorage);

    return 0;
}

int Eigenfaces::loadConfig(const map<string, string>& c)
{
    // FIXME: Because std::map has no convenient const accessor, make a copy.
    map<string, string> config(c);
    if (DEBUG)
    {
        cout<<"Load config data from a map"<<endl;
    }

    d->clearTrainingStructures();

    d->nEigens      = atoi(config["nEigens"].c_str());
    d->nTrainFaces  = atoi(config["nTrainFaces"].c_str());
    int nIds     = atoi(config["nIds"].c_str());

    d->eigenValMat  = LibFaceUtils::stringToMatrix(config["eigenValMat"],CV_32FC1);
    d->projectedTrainFaceMat = LibFaceUtils::stringToMatrix(config["projectedTrainFaceMat"],CV_32FC1);

    d->pAvgTrainImg = LibFaceUtils::stringToImage(config["avgTrainImg"], IPL_DEPTH_32F, 1);
    d->clearImages(d->eigenVectArr);
    d->eigenVectArr = std::vector<IplImage*>(d->nTrainFaces);

    //Not sure what depath and # of channels should be in faceImgArr. Store them in config?
    for ( int i = 0; i < d->nTrainFaces; ++i )
    {
        char facename[200];
        sprintf(facename, "person_%d", i);
        d->faceImgArr.push_back( LibFaceUtils::stringToImage(config[string(facename)], IPL_DEPTH_32F, 1) );
    }

    for ( int j = 0; j < d->nEigens; ++j )
    {
        char varname[200];
        sprintf(varname, "eigenVect_%d", j);
        d->eigenVectArr[j] = LibFaceUtils::stringToImage(config[string(varname)], IPL_DEPTH_32F, 1);
    }

    for ( int k = 0; k < nIds; ++k )
    {
        char idname[200];
        sprintf(idname, "indexIdMap_%d", k);
        d->indexIdMap[k] = atoi(config[string(idname)].c_str());
    }

    for ( int k = 0; k < nIds; ++k )
    {
        char idname[200];
        int id         = d->indexIdMap[k];    // Retrieve Id from the indexIdMap
        sprintf(idname, "idCountMap_%d", id);
        d->idCountMap[id] = atoi(config[string(idname)].c_str());

        //cvWrite(fileStorage, idname, (int *)&d->idCountMap[k], cvAttrList(0,0));
    }

    return 0;
}

pair<int, double> Eigenfaces::recognize(IplImage* input)
{
    if (input == 0)
    {
        if (DEBUG)
        {
            cout << "No faces passed. No recognition to do." << endl;
        }
        return make_pair<int, double>(-1, -1); // Nothing
    }

    clock_t recog;
    recog = clock();

    if (DEBUG)
    {
        cout << "Test face loaded." << endl;
    }

    //vector<int> closestIDs;

    // Now project the test images onto the PCA subspace so that comparisions can be made with trained data
    float* projectedTestFace = (float*)cvAlloc(d->nEigens * (sizeof(float)));

    // Project the test image onto the PCA subspace
    if (DEBUG)
    {
        cout << "Projecting the test image..." << endl;
    }
    cvEigenDecomposite( input,
                        d->nEigens,
                        &d->eigenVectArr[0],
                        0,
                        0,
                        d->pAvgTrainImg,
                        projectedTestFace );

    // Projected
    if (DEBUG)
    {
        cout << "Projection complete." << endl;
    }

    // Now do distance checks
    return d->findNearestNeighbor(projectedTestFace);

    //return recogResult;
}

int Eigenfaces::saveConfig(const string& dir)
{
    if (DEBUG)
        cout << "Saving config in "<< dir << endl;

    if (d->nTrainFaces == 0)
        return 0;

    string configFile          = dir + string("/libface-config.xml");
    CvFileStorage* fileStorage = cvOpenFileStorage(d->configFile.c_str(), 0, CV_STORAGE_WRITE);

    if (!fileStorage)
    {
        if (DEBUG)
            cout << "Cant open for storing :" << d->configFile << endl;

        return 1;
    }

    // Start storing
    int nIds = d->idCountMap.size();

    // Write some initial params and matrices
    cvWriteInt( fileStorage, "nEigens",               d->nEigens );
    cvWriteInt( fileStorage, "nTrainFaces",           d->nTrainFaces );
    cvWriteInt( fileStorage, "nIds",                  d->idCountMap.size() );
    if (d->eigenValMat)
        cvWrite(fileStorage, "eigenValMat",           d->eigenValMat,           cvAttrList(0,0) );
    if (d->projectedTrainFaceMat)
        cvWrite(fileStorage, "projectedTrainFaceMat", d->projectedTrainFaceMat, cvAttrList(0,0) );
    if (d->pAvgTrainImg)
        cvWrite(fileStorage, "avgTrainImg",           d->pAvgTrainImg,          cvAttrList(0,0) );


    // Write all the training faces
    for ( int i = 0; i < d->nTrainFaces; ++i )
    {
        char facename[200];
        sprintf(facename, "person_%d", i);
        cvWrite(fileStorage, facename, d->faceImgArr[i], cvAttrList(0,0));
    }

    // Write all the eigenvectors
    for ( int j = 0; j < d->nEigens; ++j )
    {
        char varname[200];
        sprintf(varname, "eigenVect_%d", j);
        cvWrite(fileStorage, varname, d->eigenVectArr[j], cvAttrList(0,0));
    }

    for ( int k = 0; k < nIds; ++k)
    {
        char idname[200];
        sprintf(idname, "indexIdMap_%d", k);
        cvWriteInt( fileStorage, idname, d->indexIdMap[k]);
    }

    for ( int k = 0; k < nIds; ++k)
    {
        char idname[200];
        int id = d->indexIdMap[k];    // Retrieve Id from the indexIdMap
        sprintf(idname, "idCountMap_%d", id);
        cvWriteInt( fileStorage, idname, d->idCountMap[id]);
    }

    // Release the fileStorage
    cvReleaseFileStorage(&fileStorage);
    return 0;
}

vector<int> Eigenfaces::update(vector<Face>& newFaceArr)
{
    vector<int> assignedIDs;
    if (newFaceArr.size() == 0)
    {
        if (DEBUG)
        {
            cout<<" No faces passed. Not training." <<endl;
        }

        return assignedIDs;
    }

    clock_t update;
    update = clock();

    unsigned int i = 0;

    // Our method : If no Id specified (-1), then give the face the next available ID.
    // Now, all ID's, when the DB is read back, were earlier read as the storage index with which the face was stored in the DB.
    // Note that indexIdMap, unlike idCountMap, is only changed when a face with a new ID is added.

    for (; i < newFaceArr.size() ; ++i)
    {
        if(newFaceArr.at(i).getId() == -1)
        {
            if (DEBUG)
            {
                cout << "Has no specified ID" << endl;
            }

            // If there was a junk image before, remove it
            if(d->idCountMap.count(-1) > 0)
            {
                cvReleaseImage(&d->faceImgArr[0]);
                d->faceImgArr.erase(d->faceImgArr.begin());
                d->nTrainFaces--;
                d->idCountMap.erase(d->idCountMap.find(-1));
                d->indexIdMap[0] = 0;
                // Nothing in index = 1 now
                d->indexIdMap.erase(d->indexIdMap.find(1));
            }

            int newId = d->faceImgArr.size();

            while( d->idCountMap.count(newId) > 1 )    // If this ID is already existing, we must acquire a new one
            {
                newId++;
            }

            // We now have the greatest unoccupied ID.
            if (DEBUG)
            {
                cout << "Giving it the ID = " << newId << endl;
            }

            d->faceImgArr.push_back(cvCloneImage(newFaceArr.at(i).getFace()));
            assignedIDs.push_back(newId);
            newFaceArr.at(i).setId(newId);

            d->nTrainFaces++;

            // A new face with a new ID is added. So map it's DB storage index with it's ID
            d->indexIdMap[d->nTrainFaces-1] = newId;
            // This a new ID added to the DB. So make an entry for it in the idCountMap
            d->idCountMap[newId]            = 1;

            //if (DEBUG)
            //    cout << "New Face added = " << d->nTrainFaces << endl;
        }
        else
        {
            int id = newFaceArr.at(i).getId();

            if (DEBUG)
            {
                cout << " Given ID as " << id << endl;
            }
            // If the ID is already in the DB

            if(d->idCountMap.count(id) > 0)
            {
                unsigned int j = 0;

                if (DEBUG)
                {
                    cout<<"Specified ID already axists in DB, averaging"<<endl;
                }

                // Must get the index of this id, so the previous face of the ID can be accessed from faceImgArr
                for(j = 0; j < d->indexIdMap.size(); ++j)
                    if(d->indexIdMap[j] == id)
                        break;

                IplImage* oldImg = d->faceImgArr.at(j);
                //cout << "SSS" << endl;
                IplImage* newImg = cvCreateImage(cvSize(oldImg->width, oldImg->height), oldImg->depth, oldImg->nChannels);
                //cout << "SSS" << endl;
                // The weight is accessed from the map
                double weight    = d->idCountMap[id];

                cvAddWeighted(oldImg, 1 - 1/weight,newFaceArr.at(i).getFace(), 1/weight, 0, newImg);

                /*
                //Cheese: Uncomment this if you want to see the average image so far for the person with this ID!
                cvNamedWindow("a");
                cvShowImage("a", newImg);
                cvWaitKey(0);
                cvDestroyWindow("a");
                */

                replace(d->faceImgArr.begin(), d->faceImgArr.end(), oldImg, newImg);
                //assignedIDs.push_back(id);
                // Increment averaging count
                d->idCountMap[id] = d->idCountMap[id] + 1;
                assignedIDs.push_back(id);

                if (DEBUG)
                {
                    cout<<"Face updated and averaged = "<<id<<endl;
                }
            }
            else
            {
                // If this is a fresh ID, and not autoassigned

                if (DEBUG)
                {
                    cout << "Specified ID does not exist in DB, so this is a new face" << endl;
                }

                // If there was a junk image before, remove it

                if(d->idCountMap.count(-1) > 0)
                {
                    cvReleaseImage(&d->faceImgArr[0]);
                    d->faceImgArr.erase(d->faceImgArr.begin());
                    d->nTrainFaces--;
                    d->idCountMap.erase(d->idCountMap.find(-1));
                    d->indexIdMap[0] = 0;
                    // Nothing in index = 1 now
                    d->indexIdMap.erase(d->indexIdMap.find(1));
                }

                d->faceImgArr.push_back(cvCloneImage(newFaceArr.at(i).getFace()));
                assignedIDs.push_back(id);
                d->nTrainFaces++;

                // A new face with a new ID is added. So map it's DB storage index with it's ID
                d->indexIdMap[d->nTrainFaces -1] = id;

                // This a new ID added to the DB. So make an entry for it in the idCountMap
                d->idCountMap[id]                = 1;
            }
        }
    }

    // DB full or all faces exhausted, now train and store
    d->learn();

    update = clock() - update;
    if (DEBUG)
    {
        printf("Updating took: %f sec.\n", (double)update / ((double)CLOCKS_PER_SEC));
        cout << "Inside Eigenfaces::update(), number of assigned ID's is " << assignedIDs.size() << endl;
    }
    return assignedIDs;
}

int Eigenfaces::count() const
{
    return d->nTrainFaces;
}

int Eigenfaces::count(int id) const
{
    return d->idCountMap[id];
}

} // namespace libface
