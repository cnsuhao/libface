/** ===========================================================
 * @file
 *
 * This file is a part of libface project
 * <a href="http://libface.sourceforge.net">http://libface.sourceforge.net</a>
 *
 * @date    2010-01-03
 * @brief   Test example.
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
#include "Face.h"
#include "LibFaceUtils.h"

using namespace std;

//Use namespace libface in the library.
using namespace libface;

int main(int argc, char** argv)
{
    cout << "=== This is Test.cpp ===" << endl;
    cout << "This binary will recognize known faces." << endl;

    if (argc < 2)
    {
        cout << "Bad Args!!!\nUsage: " << argv[0] << " <image1> <image2> ..." << endl;
        return 0;
    }

    // "." means look for configuration file in current directory

    LibFace libFace(ALL, ".");

    // This is a vector of pointers to Face objects. The vector will be destructed later. that means that the pointers will get destructed, but the faces they point to will not get deconstructed. We should destruct them manually.
    vector<Face*>* result;
    // Same holds for this object. Difference is that this vector is allocated with new here. See below to find out why.
    vector<Face*>* finalresult = new vector<Face*>;

    for (int i = 1; i < argc; ++i)
    {
        // Load input image
        cout << "Loading image " << argv[i] << endl;

        // Now here, the function libFace.detectFaces will allocated the memory for result, hence we didn't allocate it earlier.
        // (Actually, some other function called by detectFaces will do it.)
        result = libFace.detectFaces(string(argv[i])); // detect faces in image

        // output results, show every face marked with a rectangle
        cout << " Face detection completed, found " << result->size() << " faces." << endl;
        for (unsigned j = 0; j < result->size(); ++j)
        {
            cout << " Drawing face " << j+1 << "." << endl;
            IplImage* img = cvLoadImage(argv[i], CV_LOAD_IMAGE_GRAYSCALE);
            Face* face = result->at(j);
            cvRectangle( img, cvPoint(face->getX1(), face->getY1()), cvPoint(face->getX2(), face->getY2()), CV_RGB(255,0,0), 3, 2, 0);
            //LibFaceUtils::showImage(img,argv[i]);
            cvReleaseImage(&img);
        }

        // Here, we insert the pointer from result into finalresult, which is why we had to allocate it earlier. finalresult and result now point to the same Face objects! We have to be careful not to destruct them twice, or destruct them through one pointer while the other pointer still needs them.
        finalresult->insert(finalresult->end(), result->begin(), result->end());

        // Now we can deallocate result. This will call the destructor ~vector<Face*>(). This again calls the destructors for all elements of the vector, but those are just pointers. The destructor of a pointer is a no-op. meaning it does not release the object it is pointing to. Now this actually what we want, since we still need the actual object - finalresult is pointing to it. We still need to delete result, because in the next iteration of the for loop, result will be changed to point somewhere else.
        delete result;
    }

    cout << "Will recognize " << finalresult->size() << " faces..." << endl;

    vector<pair<int, float> >recognised;
    recognised = libFace.recognise(finalresult);

    cout << "Recognition done, presenting results." << endl;

    if(recognised.size() != finalresult->size()) {
        cout << "Error, size mismatch, exiting." << endl;
        return 1;
    }

    for(unsigned i = 0; i < recognised.size(); ++i)
    {
        cout << " Face No." << i+1 << " matched known face with ID " << recognised.at(i).first << " at a distance of " << recognised.at(i).second << "." << endl;
    }

    // Now that we are done, we still have finalresult pointing to some Face objects (that is, if we found any in the images). Once main() returns, the destructor for finalresult will be called automatically, which in turn calls the destructors for all the elements of the vector. However, this will only delete the pointers, but not the Face objects pointed to, as already desribed earlier. We need to manually destruct all objects that finalresult points to.
    while(finalresult->size() > 0) {
        // Deleting the pointer calls the deconstructor of the object pointed to.
        delete finalresult->at(0);
        finalresult->erase(finalresult->begin());
    }
    delete finalresult;

    cout << "=== This was Test.cpp ===" << endl;
    return 0;
}
