/** ===========================================================
 * @file
 *
 * This file is a part of libface project
 * <a href="http://libface.sourceforge.net">http://libface.sourceforge.net</a>
 *
 * @date    2010-01-03
 * @brief   Train parser example.
 * @section DESCRIPTION
 *
 * This is a simple example of the use of the libface library.
 * It implements face detection and recognition and uses the opencv libraries.
 *
 * @note: libface does not require users to have openCV knowledge, so here, 
 *        openCV is treated as a "3rd-party" library for image manipulation convenience.
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
#include <vector>

// Extra libraries for use in client programs
#if defined (__APPLE__)
#include <cv.h>
#include <highgui.h>
#else
#include <opencv/cv.h>
#include <opencv/highgui.h>
#endif

// Our library
#include "LibFace.h"
#include "LibFaceUtils.h"
#include "Face.h"

using namespace std;

//Use namespace libface in the library.
using namespace libface;

int main(int argc, char** argv)
{
    cout << "=== This is Train.cpp ===" << endl;
    cout << " This binary will memorize new faces. Usage: " << argv[0] << " <image1> <image2> ... ." << endl;
    cout << " No sanity check is done for image file paths." << endl;

    if (argc < 2)
    {
        cout << "Bad Args!!!\nUsage: " << argv[0] << " <image1> <image2> ..." << endl;
        return 0;
    }

    // Load libface with DETECT to only do detection mode
    // "." means look for configuration file in current directory
    LibFace libFace = LibFace(ALL, ".");

    // Vector of faces returned from a particular photo's detection
    vector<Face*>* result;

    // The combined vector of faces after detection on all photos is over
    vector<Face*>* finalresult = new vector<Face*>;

    for (int i = 1; i < argc; ++i)
    {
        // Load input image
        cout << "Detecting faces in image " << argv[i] << "." << endl;
        result = libFace.detectFaces(string(argv[i]));
        cout << " Face detection completed, found " << result->size() << " faces." << endl;

        // Draw squares over detected faces
        for (unsigned j = 0; j < result->size(); ++j)
        {
            cout << " Drawing face " << j+1 << "." << endl;
            IplImage* img = cvLoadImage(argv[i], CV_LOAD_IMAGE_COLOR);
            Face* face = result->at(j);
            cvRectangle( img, cvPoint(face->getX1(), face->getY1()), cvPoint(face->getX2(), face->getY2()), CV_RGB(255,0,0), 3, 2, 0);
            LibFaceUtils::showImage(img,string(argv[i]));
            cvReleaseImage(&img);
    }

        // Append result to finalresult
        finalresult->insert(finalresult->end(), result->begin(), result->end());

        // deallocate result (just calling clear() leaves us with a basic vector structure still allocated)
        delete result;
    }

    cout << "Will now train with " << finalresult->size() << " faces." << endl;

    libFace.update(finalresult);

    cout << "Training done, presenting results." << endl;

    for(unsigned i = 0; i < finalresult->size(); ++i) {
        cout << " ID "<< finalresult->at(i)->getId() << " assigned to face " << i << ", which is now being drawn."<< endl;
        stringstream title;
        title << "ID " << finalresult->at(i)->getId();
        LibFaceUtils::showImage(finalresult->at(i)->getFace(),title.str());
    }

    libFace.saveConfig(".");

    // deallocate finalresult
    delete finalresult;

    cout << "=== This was Train.cpp ===" << endl;
    return 0;
}
