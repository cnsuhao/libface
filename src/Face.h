/** ===========================================================
 * @file Face.h
 *
 * This file is a part of libface project
 * <a href="http://libface.sourceforge.net">http://libface.sourceforge.net</a>
 *
 * @date    2010-03-03
 * @brief   Class for information about a face in an image.
 * @section DESCRIPTION
 *
 * Holds information about a face. The coordinates of the box representing
 * the face and the associated ID. It is used as the output for any face detection.
 * Used as input for any face recognition.
 *
 * @author Copyright (C) 2010 by Alex Jironkin
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

#ifndef _FACE_H_
#define _FACE_H_

// LibFace headers
#include "LibFaceConfig.h"

// OpenCV headers
#if defined(__APPLE__)
#include <cv.h>
#else
#include <opencv/cv.h>
#endif

#include <string>
using namespace std;

namespace libface
{

class FACEAPI Face
{
public:

    /**
     * Default constructor for the Face class. By default everything is set to -1.
     *
     * @param x1 X coordinate of the top left corner of the face.
     * @param y1 Y coordinate of the top left corner of the face.
     * @param x2 X coordinate of the bottom right corner of the face.
     * @param y2 Y coordinate of the bottom right corner of the face.
     * @param id ID of the face. -1 not known.
     * @param face A pointer to the IplImage with the image data.
     */
    Face(int x1=-1, int y1=-1, int x2=-1, int y2=-1, int id=-1, IplImage* face=0);

    /**
     * Copy constructor.
     *
     * @param that Object to be copied.
     */
    Face(const Face& that);

    /**
     * Assignment operator.
     *
     * @param that Object to be copied.
     *
     * @return Reference to assignee.
     */
    Face& operator = (const Face& that);

    // TODO do we need these?
    // bool operator == (const Face& that) const
    // bool operator != (const Face& that) const

    /**
     * Destructor that releases the IplImage.
     */
    ~Face();

    /**
     * Sets the value for the x coordinate of the top left corner of the face.
     *
     * @param x1 X coordinates of the top left corner of the face.
     */
    void setX1(int x1);

    /**
     * Sets the value for the x coordinate of the bottom right corner of the face.
     *
     * @param x2 X coordinates of the bottom right corner of the face.
     */
    void setX2(int x2);

    /**
     * Sets the value for the y coordinate of the top left corner of the face.
     *
     * @param y1 Y coordinates of the top left corner of the face.
     */
    void setY1(int y1);

    /**
     * Sets the value for the y coordinate of the bottom right corner of the face.
     *
     * @param y2 Y coordinates of the bottom right corner of the face.
     */
    void setY2(int y2);

    /**
     * Sets the id of the face.
     * @param id ID of the face.
     */
    void setId(int id);

    /**
      * New Addition - Sets the name of the face - tag name
      * @param tag - Name of the string
      */
    void setName(string tag);

    /**
     * Sets the image of the face.
     *
     * @param face pointer to the IplImage of the face.
     */
    void setFace(IplImage* face);

    /**
     * Gets the image of the face.
     *
     * @return Pointer the IplImage image of the face.
     */
    IplImage* getFace() const;

    /**
     * Gets the height of the face rectangle.
     *
     * @return Returns the height of the face rectangle.
     */
    int getHeight() const;

    /**
     * Gets the width of the face rectangle.
     *
     * @return Returns the width of the face rectangle.
     */
    int getWidth() const;

    /**
     * Gets the x coordinate of the top left corner of the face.
     *
     * @return x coordinate of the top left corner of the face.
     */
    int getX1() const;

    /**
     * Gets the x coordinate of the bottom right corner of the face.
     *
     * @return x coordinate of the bottom right corner of the face.
     */
    int getX2() const;

    /**
     * Gets the y coordinate of the top left corner of the face.
     *
     * @return y coordinate of the top left corner of the face.
     */
    int getY1() const;

    /**
     * Gets the y coordinate of the bottom right corner of the face.
     *
     * @return y coordinate of the bottom right corner of the face.
     */
    int getY2() const;

    /**
     * Gets the ID of the face.
     *
     * @return ID of the face.
     */
    int getId() const;

    /**
      * Gets the tagname of face
      *
      * @return tag of the face
      */
    string getName() const;

private:

    class FacePriv;
    FacePriv* const d;

};

} // namespace libface

#endif /* _FACE_H_ */
