/** ===========================================================
 * @file Eigenfaces.cpp
 *
 * This file is a part of libface project
 * <a href="http://libface.sourceforge.net">http://libface.sourceforge.net</a>
 *
 * @date    2009-12-27
 * @brief   Eigenfaces parser.
 * @section DESCRIPTION
 *
 * This class is an implementation of Eigenfaces. The image stored is the projection of the faces with the
 * closest match. The relation in recognition is determined by the Eigen value when decomposed.
 *
 * @author Copyright (C) 2009-2010 by Alex Jironkin
 *         <a href="alexjironkin at gmail dot com">alexjironkin at gmail dot com</a>
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

#ifdef WIN32
// To avoid warnings from MSVC compiler about OpenCV headers
#pragma warning( disable : 4996 )
#endif // WIN32

// own header
#include "Eigenfaces.h"

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

#include <time.h>

using namespace std;

namespace libface {

class Eigenfaces::EigenfacesPriv {

public:

    /**
     * Constructor.
     */
    EigenfacesPriv();

    /**
     * Copy constructor.
     *
     * @param that Object to be copied.
     */
    EigenfacesPriv(const EigenfacesPriv& that);

    /**
     * Assignment operator.
     *
     * @param that Object to be copied.
     *
     * @return Reference to assignee.
     */
    EigenfacesPriv& operator = (const EigenfacesPriv& that);

    /**
     * Destructor.
     */
    ~EigenfacesPriv();

    /**
     * TODO
     *
     * @param img1
     * @param img2
     * @return
     *
     */
    float eigen(IplImage* img1, IplImage* img2);

    /**
     * Calculates Root Mean Squared error between 2 images. The method doesn't modify input images.
     * N.B. only 1 channel is used at present.
     *
     * @param img1 First input image to compare with.
     * @param img2 Second input image to compare with.
     *
     * @return Root mean squared error.
     */
    double rms(const IplImage* img1, const IplImage* img2);

    /**
     * Performs PCA on the current training data, projects the training faces, and stores them in a DB.
     *
     * @param index Index of the previous image to be merged with.
     * @param newFace A pointer to the new face to be merged with previous one stored at index.
     */
    void learn(int index, IplImage* newFace);

    // Face data members, stored in the DB
    // Array of face images. It is assumed that all elements of faceImgArr always point to valid IplImages. Otherwise runtime errors will occur.
    vector<IplImage*> faceImgArr;
    vector<int> indexMap;

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
    int m_no_principal_components;
    vector<Mat> m_projections;
    vector<int> m_labels;
    Mat m_eigenvectors;
    Mat m_eigenvalues;
    Mat m_mean;

    // Identifier
    Identifier idType;
};


Eigenfaces::EigenfacesPriv::EigenfacesPriv() : faceImgArr(), indexMap(), configFile(), CUT_OFF(10000000.0), UPPER_DIST(10000000), LOWER_DIST(10000000), THRESHOLD(1000000.0), RMS_THRESHOLD(10.0), FACE_WIDTH(120), FACE_HEIGHT(120) {

}

Eigenfaces::EigenfacesPriv::EigenfacesPriv(const EigenfacesPriv& that) : faceImgArr(), indexMap(that.indexMap), configFile(that.configFile), CUT_OFF(that.CUT_OFF), UPPER_DIST(that.UPPER_DIST), LOWER_DIST(that.LOWER_DIST), THRESHOLD(that.THRESHOLD), RMS_THRESHOLD(that.RMS_THRESHOLD), FACE_WIDTH(that.FACE_WIDTH), FACE_HEIGHT(that.FACE_HEIGHT) {
    // copy images pointed to by faceImgArr
    for(unsigned i = 0; i < that.faceImgArr.size(); ++i) {
        faceImgArr.push_back(cvCloneImage(that.faceImgArr.at(i)));
    }
}

Eigenfaces::EigenfacesPriv& Eigenfaces::EigenfacesPriv::operator = (const EigenfacesPriv& that) {
    LOG(libfaceWARNING) << "EigenfacesPriv& operator = : This operator has not been tested.";
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

Eigenfaces::EigenfacesPriv::~EigenfacesPriv() {
    for(unsigned i = 0; i < faceImgArr.size(); ++i) {
        cvReleaseImage(&faceImgArr.at(i));
    }
}

float Eigenfaces::EigenfacesPriv::eigen(IplImage* img1, IplImage* img2) {

    // TODO just a question, why is malloc prefered here over new?
    /*
    I tried using new and delete, resulting in 24 bytes less being allocated per function call.
    I believe this is because cvAlloc allocates more memory than actually needed.
    Using an array of pointers instead of a vector saves an additional 24 bytes per call.
    */
    // same applies to function learn

    //#define new_and_delete
#ifdef new_and_delete

    // tempfaces is also not used in this version

    float minDist = FLT_MAX;

    //#define tempFaces_as_vector
#ifdef tempFaces_as_vector
    vector<IplImage*> tempFaces;
    tempFaces.push_back(img1);
    tempFaces.push_back(img2);
#else
    IplImage * tempFaces[2];
    tempFaces[0] = img1;
    tempFaces[1] = img2;
#endif

    float* eigenValues = new float[2];

    float* projectedTestFace = new float;

    CvSize size = cvSize(img1->width, img1->height);

    //Set PCA's termination criterion
    CvTermCriteria mycrit = cvTermCriteria(CV_TERMCRIT_NUMBER, 1, 0);

    // Initialize pointer to the average image
    IplImage* pAvgTrainImg;
    // allocate it
    if(!(pAvgTrainImg = cvCreateImage( size, IPL_DEPTH_32F, 1))) {
        LOG(libfaceERROR) << "Problems initializing pAvgTrainImg...";
    }

    // Initialise pointer to the pointers with eigen objects
    IplImage** eigenObjects = new IplImage*[2];
    // allocate it
    for(int i = 0; i < 2; i++ ){
        eigenObjects[i] = cvCreateImage( size, IPL_DEPTH_32F, 1);
        if(!(eigenObjects[i])) {
            LOG(libfaceERROR) << "Problems initializing eigenObjects";
        }
    }

    // Perform PCA
#ifdef tempFaces_as_vector
    cvCalcEigenObjects(2, &tempFaces.front(), eigenObjects,
                       CV_EIGOBJ_NO_CALLBACK, 0, NULL, &mycrit, pAvgTrainImg, eigenValues);
#else
    cvCalcEigenObjects(2, tempFaces, eigenObjects,
                       CV_EIGOBJ_NO_CALLBACK, 0, NULL, &mycrit, pAvgTrainImg, eigenValues);
#endif

    // This is a simple min distance mechanism for recognition. Perhaps we should check similarity of images.
    if(eigenValues[0] < minDist) {
        minDist = eigenValues[0];
    }

    delete projectedTestFace;
    delete[] eigenValues;
    cvReleaseImage(&pAvgTrainImg);
    cvReleaseImage(&eigenObjects[0]);
    cvReleaseImage(&eigenObjects[1]);
    delete[] eigenObjects;

    return minDist;

#else

    float minDist = FLT_MAX;

    vector<IplImage*> tempFaces;
    tempFaces.push_back(img1);
    tempFaces.push_back(img2);

    // Initialize array with eigen values
    float* eigenValues;

    // allocate it
    if( !(eigenValues = (float*) cvAlloc( 2*sizeof(float) ) ) ) {
        LOG(libfaceERROR) << "Problems initializing eigenValues...";
    }

    float* projectedTestFace = (float*)malloc(sizeof(float));

    CvSize size = cvSize(tempFaces.at(0)->width, tempFaces.at(0)->height);

    //Set PCA's termination criterion
    CvTermCriteria mycrit = cvTermCriteria(CV_TERMCRIT_NUMBER,1,0);

    // Initialize pointer to the average image
    IplImage* pAvgTrainImg;
    // allocate it
    if( !(pAvgTrainImg = cvCreateImage( size, IPL_DEPTH_32F, 1) ) ) {
        LOG(libfaceERROR) << "Problems initializing pAvgTrainImg...";
    }

    // Initialise pointer to the pointers with eigen objects
    IplImage** eigenObjects = new IplImage *[2];
    // allocate it
    for(int i = 0; i < 2; i++ ){
        eigenObjects[i] = cvCreateImage( size, IPL_DEPTH_32F, 1 );
        if(!(eigenObjects[i] ) ) {
            LOG(libfaceERROR) << "Problems initializing eigenObjects";
        }
    }

    // Perform PCA
    cvCalcEigenObjects(2, &tempFaces.front(), eigenObjects,
                       CV_EIGOBJ_NO_CALLBACK, 0, NULL, &mycrit, pAvgTrainImg, eigenValues );




    // This is a simple min distance mechanism for recognition. Perhaps we should check similarity of images.

    /**
     * Scope of Improvement - 1, Change the distance mechanism -----------------------------------------------------
     */

    if(eigenValues[0] < minDist) {
        minDist = eigenValues[0];
    }

    //cvEigenDecomposite(tempFaces.at(0), nEigens, eigenObjects,
    //CV_EIGOBJ_NO_CALLBACK, NULL, pAvgTrainImg, projectedTestFace );

    //IplImage *proj = cvCreateImage(cvSize(input->width, input->height), IPL_DEPTH_8U, 1);
    //cvEigenProjection(eigenObjects, nEigens,
    //              CV_EIGOBJ_NO_CALLBACK, NULL, projectedTestFace, pAvgTrainImg, proj);

    //LibFaceUtils::showImage(proj);

    free(projectedTestFace);
    cvFree_(eigenValues);
    cvReleaseImage(&pAvgTrainImg);
    cvReleaseImage(&eigenObjects[0]);
    cvReleaseImage(&eigenObjects[1]);
    delete[] eigenObjects;

    // Calling clear is actually not necessary, tempFaces will be destructed on return.
    // The images pointed to in tempFaces are owned by the calling function and may not be released here (which clear would not do).
    //tempFaces.clear();

    return minDist;
#endif
}

double Eigenfaces::EigenfacesPriv::rms(const IplImage* img1, const IplImage* img2) {

    IplImage* temp = cvCreateImage(cvSize(img1->width, img1->height),img1->depth,img1->nChannels);

    cvSub(img1, img2, temp);

    IplImage* temp2 = cvCreateImage(cvSize(temp->width, temp->height), temp->depth, temp->nChannels);

    cvPow(temp, temp2, 2.0);

    double err = cvAvg(temp2).val[0];

    cvReleaseImage(&temp);
    cvReleaseImage(&temp2);

    return sqrt(err);
}

void Eigenfaces::EigenfacesPriv::learn(int index, IplImage* newFace) {

    int i;
    vector<IplImage*> tempFaces;

    tempFaces.push_back(newFace);
    tempFaces.push_back(faceImgArr.at(index));

    float* projectedFace = (float*)malloc(sizeof(float));

    CvSize size = cvSize(FACE_WIDTH, FACE_HEIGHT);

    //Set PCA's termination criterion
    CvTermCriteria mycrit = cvTermCriteria(CV_TERMCRIT_NUMBER, 1, 0);

    //Initialise pointer to the pointers with eigen objects
    IplImage** eigenObjects = new IplImage *[2];

    float* eigenValues;
    //Initialize array with eigen values
    if( !(eigenValues = (float*) cvAlloc( 2*sizeof(float) ) ) ) {
        LOG(libfaceERROR) << "Problems initializing eigenValues..."; }

    IplImage* pAvgTrainImg = 0;
    //Initialize pointer to the average image
    if( !(pAvgTrainImg = cvCreateImage( size, IPL_DEPTH_32F, 1) ) ) {
        LOG(libfaceERROR) << "Problems initializing pAvgTrainImg..."; }

    for(i = 0; i < 2; i++ ) {
        eigenObjects[i] = cvCreateImage( size, IPL_DEPTH_32F, 1 );
        if(!(eigenObjects[i] ) ) {
            LOG(libfaceERROR) << "Problems initializing eigenObjects"; }
    }

    //Perform PCA
    cvCalcEigenObjects(2, &tempFaces.front(), eigenObjects,
                       CV_EIGOBJ_NO_CALLBACK, 0, NULL, &mycrit, pAvgTrainImg, eigenValues );

    cvEigenDecomposite(tempFaces.at(0), 1, eigenObjects,
                       CV_EIGOBJ_NO_CALLBACK, NULL, pAvgTrainImg, projectedFace );

    IplImage *proj = cvCreateImage(size, IPL_DEPTH_8U, 1);
    cvEigenProjection(eigenObjects, 1,
                      CV_EIGOBJ_NO_CALLBACK, NULL, projectedFace, pAvgTrainImg, proj);

    //LibFaceUtils::showImage(proj);

    cvReleaseImage(&faceImgArr.at(index));
    faceImgArr.at(index) = proj;

    //free other stuff allocated above.
    cvFree_(eigenValues);
    free(projectedFace);

    cvReleaseImage(&pAvgTrainImg);
    cvReleaseImage(&eigenObjects[0]);
    cvReleaseImage(&eigenObjects[1]);
    delete[] eigenObjects;

}

Eigenfaces::Eigenfaces(const string& dir, Identifier id_type) : d(new EigenfacesPriv) {
    struct stat stFileInfo;
    d->configFile = dir + "/" + "Eigen-" + CONFIG_XML ;

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

Eigenfaces::Eigenfaces(const Eigenfaces& that) : d(that.d ?  new EigenfacesPriv(*that.d) : 0) {
    if(!d) {
        LOG(libfaceERROR) << "Eigenfaces(const Eigenfaces& that) : d points to NULL.";
    }
}

Eigenfaces& Eigenfaces::operator = (const Eigenfaces& that) {
    LOG(libfaceWARNING) << "Eigenfaces::operator = (const Eigenfaces& that) : This operator has not been tested.";
    if(this == &that) {
        return *this;
    }
    if( (that.d == 0) || (d == 0) ) {
        LOG(libfaceERROR) << "Eigenfaces::operator = (const Eigenfaces& that) : d or that.d points to NULL.";
    } else {
        *d = *that.d;
    }
    return *this;
}

Eigenfaces::~Eigenfaces() {
    delete d;
}

int Eigenfaces::count() const {
    return d->faceImgArr.size();
}

map<string, string> Eigenfaces::getConfig() {
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

int Eigenfaces::loadConfig(const string& dir) {
    d->configFile = dir + "/" + "Eigen-" + CONFIG_XML ;

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
//    LibFaceUtils::printMatrix(d->projectedTrainFaceMat);

    // If m_projections or m_labels are not empty make them empty
    while(d->m_projections.size()) d->m_projections.pop_back();
    while(d->m_labels.size()) d->m_labels.pop_back();

    cout << "projection size: " << d->m_projections.size() << endl;
    cout << "label size: " << d->m_labels.size() << endl;

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

int Eigenfaces::loadConfig(const map<string, string>& c) {
    // TODO FIXME: Because std::map has no convenient const accessor, make a copy.
    map<string, string> config(c);

    LOG(libfaceINFO) << "Load config data from a map.";

    int nIds  = atoi(config["nIds"].c_str()), i;

    // If m_projections or m_labels are not empty make them empty
    while(d->m_projections.size()) d->m_projections.pop_back();
    while(d->m_labels.size()) d->m_labels.pop_back();

    // Not sure what depth and # of channels should be in faceImgArr. Store them in config?
    for ( i = 0; i < nIds; i++ ) {
        char facename[200];
        sprintf(facename, "person_%d", i);
        IplImage* tmp = LibFaceUtils::stringToImage(config[string(facename)], IPL_DEPTH_32F, 1);
        d->m_projections.push_back(cvarrToMat(tmp));

        //d->faceImgArr.push_back( LibFaceUtils::stringToImage(config[string(facename)], IPL_DEPTH_32F, 1) );
    }

    for ( i = 0; i < nIds; i++ ) {
        char idname[200];
        sprintf(idname, "id_%d", i);
        d->m_labels.push_back(atoi(config[string(idname)].c_str()));
//        d->indexMap.push_back( atoi(config[string(idname)].c_str()));
    }

    return 0;
}

pair<int, float> Eigenfaces::recognize(IplImage* input) {
    if (input == 0) {
        LOG(libfaceWARNING) << "No faces passed. No recognition to do." << endl;

        return make_pair<int, float>(-1, -1); // Nothing
    }

    float minDist = FLT_MAX;
    int id = -1;
    clock_t recog = clock();
    size_t j;

    for( j = 0; j < d->faceImgArr.size(); j++) {

        float err = d->eigen(input, d->faceImgArr.at(j));

        if(err < minDist) {
            minDist = err;
            id = j;
        }
    }

    recog = clock() - recog;

    LOG(libfaceDEBUG) << "Recognition took: " << (double)recog / ((double)CLOCKS_PER_SEC) << "sec.";

    cout << "Distance: " << minDist << endl;

    if(minDist > d->THRESHOLD) {

        LOG(libfaceDEBUG) << "The value of minDist (" << minDist << ") is above the threshold (" << d->THRESHOLD << ").";

        id = -1;
        minDist = -1;

    } else
        LOG(libfaceDEBUG) << "The value of minDist is: " << minDist;


    return make_pair<int, float>(id, minDist);
}


/**
  * New Addition
  */
static Mat convertToRowMatrix(InputArray src, int matrix_type, double alpha=1, double beta=0)
{
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


/**********************************************************************************/
void Eigenfaces::training(vector<Face*>* faces, int no_principal_components){

    clock_t update;
    update = clock();

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

    int n = data.rows;

    if(n != labels.size()) {
        string error_message = format("The number of samples (src) must equal the number of labels (labels). Was len(samples)=%d, len(labels)=%d.", n, labels.size());
        error(cv::Exception(CV_StsBadArg, error_message,  "cv::Eigenfaces::train", __FILE__, __LINE__));
    }

    assert(no_principal_components >= 0);
    no_principal_components > n ? no_principal_components = n : true;

    // calculate PCA
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, no_principal_components);

    // copy the PCA results
    d->m_mean = pca.mean.reshape(1,1); // store the mean vector
    d->m_eigenvalues = pca.eigenvalues.clone(); // eigenvalues by row
    transpose(pca.eigenvectors, d->m_eigenvectors); // eigenvectors by column
    d->m_labels = labels; // store labels for prediction

    // save projections
    for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++){
        Mat p = subspaceProject(d->m_eigenvectors, d->m_mean, data.row(sampleIdx));
        d->m_projections.push_back(p);
    }

    update = clock() - update;
    printf("Whole Process took: %f sec.\n", (double)update / ((double)CLOCKS_PER_SEC));

    cout << "Projection Size: " << d->m_projections.size() << endl;
    cout << "Eigenface - Training Done " << endl;
}

/**
 * New Addition
 */
int Eigenfaces::testing(IplImage *img){

    cout << "in eigenfaces::testing --------------------------" << endl;

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

int Eigenfaces::saveConfig(const string& dir) {
    LOG(libfaceINFO) << "Saving config in "<< dir;

    // string configFile          = dir + "/" + CONFIG_XML;
    CvFileStorage* fileStorage = cvOpenFileStorage(d->configFile.c_str(), 0, CV_STORAGE_WRITE);

    if (!fileStorage) {
        LOG(libfaceERROR) << "Can't open file for storing :" << d->configFile << ". Save has failed!.";
        return 1;
    }

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

        // Writing Projection Data
        IplImage tmp = d->m_projections.at(i);
        cvWrite(fileStorage, facename, &tmp, cvAttrList(0,0));

        //Need to write eigenvector and mean also
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

int Eigenfaces::update(vector<Face*>* newFaceArr) {
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

                d->learn(*it, cvCloneImage(newFaceArr->at(i)->getFace()));

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
