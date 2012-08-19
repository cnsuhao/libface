/** ===========================================================
 * @file
 *
 * This file is a part of libface project
 * <a href="http://libface.sourceforge.net">http://libface.sourceforge.net</a>
 *
 * @date    2010-10-02
 * @brief   main window.
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

#include "mainwindow.h"
#include "ui_mainwindow.h"

// OpenCV headers
#if defined (__APPLE__)
#include <cv.h>
#include <highgui.h>
#else
#include <opencv/cv.h>
#include <opencv/highgui.h>
#endif

using namespace std;
using namespace libface;

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent),
      ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //this->connect(ui->openImageBtn, SIGNAL(clicked()), this, SLOT(openImage()));
    this->connect(ui->openImageBtn, SIGNAL(clicked()), this, SLOT(Training()));
    //this->connect(ui->openConfigBtn, SIGNAL(clicked()), this, SLOT(openConfig()));
    //this->connect(ui->detectFacesBtn, SIGNAL(clicked()), this, SLOT(detectFaces()));
    this->connect(ui->detectFacesBtn, SIGNAL(clicked()), this, SLOT(Testing()));
    this->connect(ui->recogniseBtn, SIGNAL(clicked()), this, SLOT(recognise()));
    this->connect(ui->updateDatabaseBtn, SIGNAL(clicked()), this, SLOT(updateConfig()));
    this->connect(ui->saveConfigBtn, SIGNAL(clicked()), this, SLOT(saveConfig()));

    myScene = new QGraphicsScene();

    QHBoxLayout* layout = new QHBoxLayout;
    myView              = new QGraphicsView(myScene);
    layout->addWidget(myView);

    ui->widget->setLayout(layout);

    myView->setRenderHints(QPainter::Antialiasing);
    myView->show();


//    libFace = new LibFace(libface::FISHER,QDir::currentPath().toStdString());
//    libFace = new LibFace(libface::HMM,QDir::currentPath().toStdString());
    libFace = new LibFace(libface::EIGEN,QDir::currentPath().toStdString());

    ui->configLocation->setText(QDir::currentPath());

    getTrainigData();
    getTestData();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::changeEvent(QEvent* e)
{
    QMainWindow::changeEvent(e);
    switch (e->type())
    {
    case QEvent::LanguageChange:
        ui->retranslateUi(this);
        break;
    default:
        break;
    }
}

/* void MainWindow::openImage()
{
    QString file = QFileDialog::getOpenFileName(this,
            tr("Open Image"), QDir::currentPath(), tr("Image Files (*.png *.jpg *.bmp *.pgm)"));

    clearScene();

    currentPhoto   = string(file.toAscii().data());
    QPixmap* photo = new QPixmap(file);
    lastPhotoItem  = new QGraphicsPixmapItem(*photo);

    if(1.*ui->widget->width()/photo->width() < 1.*ui->widget->height()/photo->height())
    {

        scale = 1.*ui->widget->width()/photo->width();
    }
    else
    {
        scale = 1.*ui->widget->height()/photo->height();
    }

    lastPhotoItem->setScale(scale);

    myScene->addItem(lastPhotoItem);
} */

/* void MainWindow::detectFaces()
{
    int i;
    currentFaces = libFace->detectFaces(currentPhoto);

    if(!currentPhoto.length()) return;

    int size     = currentFaces->size();

    for (i=0 ; i<size ; i++)
    {
        Face* face          = currentFaces->at(i);
        FaceItem* faceItem = new FaceItem(0,
                                          myScene, face->getX1()*scale,
                                          face->getY1()*scale,
                                          (face->getX2()-face->getX1())*scale,
                                          (face->getY2()-face->getY1())*scale);

        //cout << "Face:\t(" << face.getX1()*scale << ","<<face.getY1()*scale <<")" <<endl;
    }
} */

void MainWindow::getTrainigData(){

    cout << endl << "Training Data ---------------- " << endl << endl;
    /**
     * Select a folder consisting of input images labeled in folders
     */

    QString dir("/home/mahfuz/Coding/Face_Recognition/Libface-git/Edit/libface/examples/database/train");
    QStringList filters;
    QDir myDir(dir);
    QStringList folderList =  myDir.entryList (filters);

    cout << "Dir: " << dir.toAscii().data() << endl << " Files: " << endl;

    int count = 0;
    filters << "*.png" << "*.jpg" << "*.pgm";

    QStringList fileList;
    QString imageFile;

    currentFaces = new vector<Face*>();
    IplImage* img;
    Face* face;

    foreach (const QString &str, folderList) {
        count ++;
        if(count < 3){
            continue;
        }

        fileList.clear();

        cout << str.toAscii().data() << endl;

        // get input images from individual folders
        myDir.setPath(dir + "/" + str);
        fileList <<  myDir.entryList (filters);
        //cout << "size: " << fileList.size() << endl;


        // Now, we will get the id from the foldername
        QString tmpStr(str);
        tmpStr.remove(0,1);

        int id = tmpStr.toInt();

        cout << "ID: " << id << endl;


        // Creating the vector of faces
        foreach (const QString &filename, fileList){
            cout << count - 2 << ": " << filename.toAscii().data() << endl;

            // Store the image in a IplImage
            imageFile = dir + "/" + str + "/" + filename;

            img = cvLoadImage(imageFile.toAscii().data(), CV_LOAD_IMAGE_GRAYSCALE);
            //cout << "Width:" << img->roi->width << endl;

            face = new Face(0,0,img->width,img->height);
            face->setFace(img);
            face->setId(id);

            currentFaces->push_back(face);
        }

    }

    cout << "Total Faces: " << currentFaces->size() << endl;
}

void MainWindow::getTestData(){

    cout << endl << "Testing Data ---------------------- " << endl << endl;

    QString dir("/home/mahfuz/Coding/Face_Recognition/Libface-git/Edit/libface/examples/database/test");
    QStringList filters;

    filters << "*.png" << "*.jpg" << "*.pgm";

    QDir myDir(dir);
    QStringList folderList =  myDir.entryList (filters);

    cout << "Dir: " << dir.toAscii().data() << endl << " Files: " << endl;

    QString imageFile;

    testFaces = new vector<Face*>();
    IplImage* img;
    Face* face;

    foreach (const QString &str, folderList) {

        //cout << str.toAscii().data() << endl;

        // Store the image in a IplImage
        imageFile = dir + "/" + str;

        img = cvLoadImage(imageFile.toAscii().data(), CV_LOAD_IMAGE_GRAYSCALE);

        face = new Face(0,0,img->width,img->height);
        face->setFace(img);

        testFaces->push_back(face);
    }

}


void MainWindow::Training()
{

    cout << "Training Starts -----------" << endl;
    libFace->training(currentFaces,1);

}


void MainWindow::Testing()
{
    QString dir("/home/mahfuz/Coding/Face_Recognition/Libface-git/Edit/libface/examples/database/test");
    QStringList filters;
    filters << "*.png" << "*.jpg" << "*.pgm";
    QDir myDir(dir);
    QStringList folderList =  myDir.entryList (filters);

    cout << "Testing Starts " << endl;

    vector<int> result = libFace->testing(testFaces);

    cout << "Testing Done" << endl;

    int total = testFaces->size();

    cout << "Size: " << total << endl;

    for (int i = 0 ; i < total ; i++ ){
        cout << folderList.at(i).toAscii().data() << " -> " << result.at(i) << endl;
    }
}

void MainWindow::recognise()
{
    cout << "Load Config Called" << endl;
    libFace->loadConfig(QDir::currentPath().toStdString());

    return;
}

void MainWindow::openConfig()
{
    QString directory = QFileDialog::getExistingDirectory(this,tr("Select Config Directory"),QDir::currentPath());

    ui->configLocation->setText(directory);

    libFace = new LibFace(ALL,directory.toStdString());
}

void MainWindow::updateConfig() {
}

void MainWindow::clearScene() {
    QList<QGraphicsItem*> list = myScene->items();

    int i;

    for(i=0;i<list.size();i++)
    {
        myScene->removeItem(list.at(i));
    }
}

void MainWindow::saveConfig()
{
    //libFace->loadConfig(libFace->getConfig());
    libFace->saveConfig(QDir::currentPath().toStdString());
}
