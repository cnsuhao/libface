/** ===========================================================
 * @file Fisherfaces.cpp
 *
 * This file is a part of libface project
 * <a href="http://libface.sourceforge.net">http://libface.sourceforge.net</a>
 *
 * @date    2009-12-27
 * @brief   Fisherfaces Main File.
 * @section DESCRIPTION
 *
 * This class is an implementation of Fisherfaces algorithm.

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
#ifdef WIN32
// To avoid warnings from MSVC compiler about OpenCV headers
#pragma warning( disable : 4996 )
#endif // WIN32

// own header
#include "FisherFaces.h"

// LibFace headers
#include "Log.h"
#include "LibFaceConfig.h"
#include "Face.h"
#include "FaceDetect.h"
#include "LibFaceUtils.h"

// OpenCV headers
#if defined (__APPLE__)
#include <cvaux.h>
#else
#include <opencv/cvaux.h>
#endif

// C headers
#include <sys/stat.h>
#include <iostream>
#include <algorithm>

using namespace std;

namespace libface {

class Fisherfaces::FisherfacesPriv {

public:

    /**
     * Constructor.
     */
    FisherfacesPriv();

    /**
     * Copy constructor.
     *
     * @param that Object to be copied.
     */
    FisherfacesPriv(const FisherfacesPriv& that);

    /**
     * Assignment operator.
     *
     * @param that Object to be copied.
     *
     * @return Reference to assignee.
     */
    FisherfacesPriv& operator = (const FisherfacesPriv& that);

    /**
     * Destructor.
     */
    ~FisherfacesPriv();


    // Face data members, stored in the DB
    // Array of face images. It is assumed that all elements of faceImgArr always point to valid IplImages. Otherwise runtime errors will occur.
    vector<IplImage*> faceImgArr;
    vector<int> indexMap;
    vector<string> tagMap;

    // Config data members
    string configFile;

    double CUT_OFF;
    double UPPER_DIST;
    double LOWER_DIST;
    float THRESHOLD;
    float RMS_THRESHOLD;
    int FACE_WIDTH;
    int FACE_HEIGHT;


    /**
     * New Addition
     */
    int m_no_components_after_lda;
    vector<Mat> m_projections;
    vector<int> m_labels;
    Mat m_eigenvectors;
    Mat m_eigenvalues;
    Mat m_mean;

    // Identifier
    Identifier idType;
};


Fisherfaces::FisherfacesPriv::FisherfacesPriv() : faceImgArr(), indexMap(), configFile(), CUT_OFF(10000000.0), UPPER_DIST(10000000), LOWER_DIST(10000000), THRESHOLD(1000000.0), RMS_THRESHOLD(10.0), FACE_WIDTH(120), FACE_HEIGHT(120) {

}

Fisherfaces::FisherfacesPriv::FisherfacesPriv(const FisherfacesPriv& that) : faceImgArr(), indexMap(that.indexMap), configFile(that.configFile), CUT_OFF(that.CUT_OFF), UPPER_DIST(that.UPPER_DIST), LOWER_DIST(that.LOWER_DIST), THRESHOLD(that.THRESHOLD), RMS_THRESHOLD(that.RMS_THRESHOLD), FACE_WIDTH(that.FACE_WIDTH), FACE_HEIGHT(that.FACE_HEIGHT) {
    // copy images pointed to by faceImgArr
    for(unsigned i = 0; i < that.faceImgArr.size(); ++i) {
        faceImgArr.push_back(cvCloneImage(that.faceImgArr.at(i)));
    }
}

Fisherfaces::FisherfacesPriv& Fisherfaces::FisherfacesPriv::operator = (const FisherfacesPriv& that) {
    LOG(libfaceWARNING) << "FisherfacesPriv& operator = : This operator has not been tested.";
    if(this == &that) {
        return *this;
    }

    // release old images
    for(unsigned i = 0; i < faceImgArr.size(); ++i) {
        cvReleaseImage(&faceImgArr.at(i));
    }
    faceImgArr.clear();

    // clone new images
    for(unsigned i = 0; i < that.faceImgArr.size(); ++i) {
        faceImgArr.push_back(cvCloneImage(that.faceImgArr.at(i)));
    }

    indexMap = that.indexMap;
    configFile = that.configFile;
    CUT_OFF = that.CUT_OFF;
    UPPER_DIST = that.UPPER_DIST;
    LOWER_DIST = that.LOWER_DIST;
    THRESHOLD = that.THRESHOLD;
    RMS_THRESHOLD = that.RMS_THRESHOLD;
    FACE_WIDTH = that.FACE_WIDTH;
    FACE_HEIGHT = that.FACE_HEIGHT;

    return *this;
}

Fisherfaces::FisherfacesPriv::~FisherfacesPriv() {
    for(unsigned i = 0; i < faceImgArr.size(); ++i) {
        cvReleaseImage(&faceImgArr.at(i));
    }
}


Fisherfaces::Fisherfaces(const string& dir, Identifier id_type) : d(new FisherfacesPriv) {
    struct stat stFileInfo;
    d->configFile = dir + "/" + "Fisher-" + CONFIG_XML ;

    // Identifier assignemnt
    d->idType = id_type;

    LOG(libfaceINFO) << "Config location: " << d->configFile;

    int intStat = stat(d->configFile.c_str(),&stFileInfo);
    if (intStat == 0) {
        LOG(libfaceINFO) << "libface config file exists. Loading previous config.";
        loadConfig(dir);
    } else {
        LOG(libfaceINFO) << "libface config file does not exist. Will create new config.";
    }
}

Fisherfaces::Fisherfaces(const Fisherfaces& that) : d(that.d ?  new FisherfacesPriv(*that.d) : 0) {
    if(!d) {
        LOG(libfaceERROR) << "Fisherfaces(const Fisherfaces& that) : d points to NULL.";
    }
}

Fisherfaces& Fisherfaces::operator = (const Fisherfaces& that) {
    LOG(libfaceWARNING) << "Fisherfaces::operator = (const Fisherfaces& that) : This operator has not been tested.";
    if(this == &that) {
        return *this;
    }
    if( (that.d == 0) || (d == 0) ) {
        LOG(libfaceERROR) << "Fisherfaces::operator = (const Fisherfaces& that) : d or that.d points to NULL.";
    } else {
        *d = *that.d;
    }
    return *this;
}

Fisherfaces::~Fisherfaces() {
    delete d;
}

int Fisherfaces::count() const {
    return d->faceImgArr.size();
}

map<string, string> Fisherfaces::getConfig() {
    map<string, string> config;

    config["nIds"] = d->faceImgArr.size();
    //config["FACE_WIDTH"] = sprintf(value, "%d",d->indexMap.at(i));;

    for ( unsigned int i = 0; i < d->faceImgArr.size(); i++ ) {
        char facename[200];
        sprintf(facename, "person_%d", i);
        config[string(facename)] = LibFaceUtils::imageToString(d->faceImgArr.at(i));
    }

    for ( unsigned int i = 0; i < d->indexMap.size(); i++ ) {
        char facename[200];
        sprintf(facename, "id_%d", i);
        char value[10];
        config[string(facename)] = sprintf(value, "%d",d->indexMap.at(i));
    }

    return config;
}

int Fisherfaces::loadConfig(const string& dir) {
    d->configFile = dir + "/" + "Fisher-" + CONFIG_XML ;

    LOG(libfaceDEBUG) << "Load training data" << endl;

    CvFileStorage* fileStorage = cvOpenFileStorage(d->configFile.data(), 0, CV_STORAGE_READ);

    if (!fileStorage) {
        LOG(libfaceERROR) << "Can't open config file for reading :" << d->configFile;
        return 1;
    }

    int nIds = cvReadIntByName(fileStorage, 0, "nIds", 0), i;

    d->FACE_WIDTH = cvReadIntByName(fileStorage, 0, "FACE_WIDTH",d->FACE_WIDTH);
    d->FACE_HEIGHT = cvReadIntByName(fileStorage, 0, "FACE_HEIGHT",d->FACE_HEIGHT);
    d->THRESHOLD = cvReadRealByName(fileStorage, 0, "THRESHOLD", d->THRESHOLD);
    //LibFaceUtils::printMatrix(d->projectedTrainFaceMat);

    // If m_projections or m_labels are not empty make them empty
    while(d->m_projections.size()) d->m_projections.pop_back();
    while(d->m_labels.size()) d->m_labels.pop_back();

    for ( i = 0; i < nIds; i++ ) {
        char facename[200];
        sprintf(facename, "person_%d", i);
        IplImage* tmp = (IplImage*)cvReadByName(fileStorage, 0, facename, 0);
        d->m_projections.push_back(cvarrToMat(tmp));
    }

    char eigen_name[20];
    sprintf(eigen_name,"eigenvector");
    IplImage* eigen_tmp = (IplImage*)cvReadByName(fileStorage, 0, eigen_name, 0);
    d->m_eigenvectors = cvarrToMat(eigen_tmp);

    char mean_name[20];
    sprintf(mean_name,"mean");
    IplImage* mean_tmp = (IplImage*)cvReadByName(fileStorage, 0, mean_name, 0);
    d->m_mean = cvarrToMat(mean_tmp);

    for ( i = 0; i < nIds; i++ ) {
        char idname[200];
        sprintf(idname, "id_%d", i);
       d->m_labels.push_back(cvReadIntByName(fileStorage, 0, idname, 0));
    }

    // Release file storage
    cvReleaseFileStorage(&fileStorage);

    return 0;
}

int Fisherfaces::loadConfig(const map<string, string>& c) {
    // TODO FIXME: Because std::map has no convenient const accessor, make a copy.
    map<string, string> config(c);

    LOG(libfaceINFO) << "Load config data from a map.";

    int nIds  = atoi(config["nIds"].c_str()), i;

    // Not sure what depth and # of channels should be in faceImgArr. Store them in config?
    for ( i = 0; i < nIds; i++ ) {
        char facename[200];
        sprintf(facename, "person_%d", i);
        d->faceImgArr.push_back( LibFaceUtils::stringToImage(config[string(facename)], IPL_DEPTH_32F, 1) );
    }

    for ( i = 0; i < nIds; i++ ) {
        char idname[200];
        sprintf(idname, "id_%d", i);
        d->indexMap.push_back( atoi(config[string(idname)].c_str()));
    }

    return 0;
}

pair<int, float> Fisherfaces::recognize(IplImage* input) {

    return make_pair<int, float>(-1, -1); // Nothing
}


static Mat convertToRowMatrix(InputArray src, int matrix_type, double alpha=1, double beta=0){
    // number of samples in the InputArray
    int num_of_samples = (int) src.total();

    // if there is no data in InputArray, return an empty matrix
    if(num_of_samples == 0) return Mat();

    // dimensionality of samples
    int dimension = (int)src.getMat(0).total();

    // create data matrix
    Mat data(num_of_samples, dimension, matrix_type);

    // copy data
    for(int i = 0; i < num_of_samples; i++) {
        Mat x_i = data.row(i);
        src.getMat(i).reshape(1, 1).convertTo(x_i, matrix_type, alpha, beta);
    }

    return data;
}

/**
 * Find total number of classes - Needed for Fisherface
 */
int cmp(int i, int j){return i<j;}

int total_indentical_elements(vector <int> lbl){

    sort(lbl.begin(),lbl.end(),cmp);

    vector<int>::iterator it;
    int count = 0, value;

    //cout << "Sorted Vector:" << endl;

    for (it = lbl.begin(); it!= lbl.end() ; ++it){
        //cout << " " << *it;
        value = *it;
        while (*it == value && it!= lbl.end()){
            ++it;
        }
        count++;
        if(it == lbl.end()) break;
    }
//    cout << endl;

    return count;
}

/**********************************************************************************/
void Fisherfaces::training(vector<Face*>* faces, int no_principal_components){

    vector<Mat> src;
    vector<int> labels;

    int size = faces->size();
    for (int i = 0 ; i < size ; i++) {

        Face* face = faces->at(i);

        const IplImage* faceImg = face->getFace();

        src.push_back(cvarrToMat(faceImg));
        labels.push_back(face->getId());
    }

    // No face to process
    if(faces->size() == 0 ){
        cout << "Training Data is Empty ... can't proceed" << endl;
        exit(0);
    }

    Mat calc = src.at(0);
    d->FACE_WIDTH = calc.rows;
    d->FACE_HEIGHT = calc.cols;

    Mat data = convertToRowMatrix(src, CV_64FC1);
    int N = data.rows;

    if(labels.size() != (size_t)N)
        CV_Error(CV_StsUnsupportedFormat, "Labels must be given as integer (CV_32SC1).");

    // compute number of classes in the training images.
    int C = total_indentical_elements(labels);

    d->m_no_components_after_lda = (C-1);


    // Initially feature space is of size  width*height of image by N(# of images)
    // After doing a PCA the feature space reduces to N-C by N
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, (N-C));

    // We run LDA on the reduced feature space and final space redues to N-C by m (max of m = C-1)
    LDA lda(pca.project(data),labels, d->m_no_components_after_lda);

    d->m_mean = pca.mean.reshape(1,1);
    d->m_labels = labels;

    // store the eigenvalues of the discriminants
    lda.eigenvalues().convertTo(d->m_eigenvalues, CV_64FC1);

    // Now we calculate the total projection matrix by multiplying eigenvector of PCA with eigenvector of LDA
    gemm(pca.eigenvectors, lda.eigenvectors(), 1.0, Mat(), 0.0, d->m_eigenvectors, CV_GEMM_A_T);

    // save projections
    for(int i = 0; i < data.rows; i++) {
        Mat p = subspaceProject(d->m_eigenvectors, d->m_mean, data.row(i));
        d->m_projections.push_back(p);
    }

    cout << "Fisherface - Training Done " << endl;
}


int Fisherfaces::testingID(IplImage *img){

    Mat test = cvarrToMat(img);
    Mat q = subspaceProject(d->m_eigenvectors, d->m_mean, test.reshape(1,1));
    double minDist = DBL_MAX;
    int outputClass = -1;

    for(int i = 0; i < d->m_projections.size(); i++) {

        double distance = norm(d->m_projections[i], q, NORM_L2);

        if(distance < minDist) {
            minDist = distance;
            outputClass = d->m_labels[i];
        }
    }

    return outputClass;
}

int Fisherfaces::saveConfig(const string& dir) {
    LOG(libfaceINFO) << "Saving config in "<< dir;

    // string configFile          = dir + "/" + CONFIG_XML;
    CvFileStorage* fileStorage = cvOpenFileStorage(d->configFile.c_str(), 0, CV_STORAGE_WRITE);

    if (!fileStorage) {
        LOG(libfaceERROR) << "Can't open file for storing :" << d->configFile << ". Save has failed!.";

        return 1;
    }

    // Start storing
    //unsigned int nIds = d->faceImgArr.size(), i;

    unsigned int nIds = d->m_projections.size(), i;
    cout << "Total: " << nIds << endl;

    // Write some initial params and matrices
    cvWriteInt( fileStorage, "nIds", nIds );
    cvWriteInt( fileStorage, "FACE_WIDTH", d->FACE_WIDTH);
    cvWriteInt( fileStorage, "FACE_HEIGHT", d->FACE_HEIGHT);
    cvWriteReal( fileStorage, "THRESHOLD", d->THRESHOLD);

    // Write all the training faces
    for ( i = 0; i < nIds; i++ ) {
        char facename[200];
        sprintf(facename, "person_%d", i);
        IplImage tmp = d->m_projections.at(i);
        cvWrite(fileStorage, facename, &tmp, cvAttrList(0,0));
    }

    //Writing the whole eigenface and mean makes it infeasible as the filesize can be huge
    char eigen_name[20];
    sprintf(eigen_name,"eigenvector");
    IplImage eigen_tmp = d->m_eigenvectors;
    cvWrite(fileStorage, eigen_name, &eigen_tmp, cvAttrList(0,0));

    char mean_name[20];
    sprintf(mean_name,"mean");
    IplImage mean_tmp = d->m_mean;
    cvWrite(fileStorage, mean_name, &mean_tmp, cvAttrList(0,0));

    for ( i = 0; i < nIds; i++ ) {
        char idname[200];
        sprintf(idname, "id_%d", i);
        cvWriteInt(fileStorage, idname, d->m_labels.at(i));
    }

    // Release the fileStorage
    cvReleaseFileStorage(&fileStorage);
    return 0;
}

int Fisherfaces::update(vector<Face*>* newFaceArr) {
    if (newFaceArr->size() == 0) {
        LOG(libfaceWARNING) << " No faces passed. Not training.";

        return 0;
    }

    clock_t update;
    update = clock();

    // Our method : If no Id specified (-1), then give the face the next available ID.
    // Now, all ID's, when the DB is read back, were earlier read as the storage index with which the face was stored in the DB.
    // Note that indexIdMap, unlike idCountMap, is only changed when a face with a new ID is added.
    for (unsigned i = 0; i < newFaceArr->size() ; ++i) {
        if(newFaceArr->at(i)->getId() == -1) {
            LOG(libfaceDEBUG) << "Has no specified ID.";

            int newId = d->faceImgArr.size();

            // We now have the greatest unoccupied ID.
            LOG(libfaceDEBUG) << "Giving it the ID = " << newId;

            d->faceImgArr.push_back(cvCloneImage(newFaceArr->at(i)->getFace()));
            newFaceArr->at(i)->setId(newId);

            // A new face with a new ID is added. So map it's DB storage index with it's ID
            //d->indexIdMap[newId] = newId;
            d->indexMap.push_back(newId);
        } else {
            int id = newFaceArr->at(i)->getId();

            LOG(libfaceDEBUG) << " Given ID as " << id;

            //find (d->indexMap.begin(), d->indexMap.end(), id);

            vector<int>::iterator it = find(d->indexMap.begin(), d->indexMap.end(), id);//d->indexMap.
            if(it != d->indexMap.end()) {

                LOG(libfaceDEBUG) << "Specified ID already exists in the DB, merging 2 together.";

                //d->learn(*it, cvCloneImage(newFaceArr->at(i)->getFace()));

            } else {
                // If this is a fresh ID, and not autoassigned
                LOG(libfaceDEBUG) << "Specified ID does not exist in the DB, creating new face.";

                d->faceImgArr.push_back(cvCloneImage(newFaceArr->at(i)->getFace()));
                // A new face with a new ID is added. So map it's DB storage index with it's ID
                d->indexMap.push_back(id);
            }
        }
    }

    update = clock() - update;

    LOG(libfaceDEBUG) << "Updating took: " << (double)update / ((double)CLOCKS_PER_SEC) << "sec.";

    return 0;
}

} // namespace libface
