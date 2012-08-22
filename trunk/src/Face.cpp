/** ===========================================================
 * @file Face.cpp
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

// own header
#include "Face.h"

// LibFace headers
#include "Log.h"

namespace libface {

class Face::FacePriv {

public:

    /**
     * Default constructor for the FacePriv class. By default everything is set to -1.
     *
     * @param x1 X coordinate of the top left corner of the face.
     * @param y1 Y coordinate of the top left corner of the face.
     * @param x2 X coordinate of the bottom right corner of the face.
     * @param y2 Y coordinate of the bottom right corner of the face.
     * @param id ID of the face. -1 not known.
     * @param face A pointer to the IplImage with the image data.
     */
    FacePriv(int x1=-1, int y1=-1, int x2=-1, int y2=-1, int id=-1, IplImage* face=0);

    /**
     * Copy constructor.
     *
     * @param that Object to be copied.
     */
    FacePriv(const FacePriv& that);

    /**
     * Assignment operator.
     *
     * @param that Object to be copied.
     *
     * @return Reference to assignee.
     */
    FacePriv& operator = (const FacePriv& that);

    /**
     * Destructor that releases the IplImage.
     */
    ~FacePriv();

    /**
     * Manually release the data held in the face object.
     */
    void releaseData();

    int             x1;
    int             y1;
    int             x2;
    int             y2;
    int             id;
    string          tagName;
    int             width;
    int             height;
    IplImage*       face;
};


Face::FacePriv::FacePriv(int x1, int y1, int x2, int y2, int id, IplImage* face) : x1(x1), y1(y1), x2(x2), y2(y2), id(id), width(x2-x1), height(y2-y1), face(face) {}

Face::FacePriv::FacePriv(const FacePriv& that) : x1(that.x1), y1(that.y1), x2(that.x2), y2(that.y2), id(that.id), width(that.width), height(that.height), face(0) {
    if(that.face) {
        face = cvCloneImage(that.face);
    }
}

Face::FacePriv& Face::FacePriv::operator = (const FacePriv& that) {
    if(this == &that) {
        return *this;
    }
    x1 = that.x1;
    y1 = that.y1;
    x2 = that.x2;
    y2 = that.y2;
    id = that.id;
    width = that.width;
    height = that.height;
    if(face) {
        cvReleaseImage(&face);
    }
    if(that.face) {
        face = cvCloneImage(that.face);
    }
    return *this;
}

Face::FacePriv::~FacePriv() {
	if(this->face) {
		cvReleaseImage(&this->face);
	}
}

void Face::FacePriv::releaseData() {
	LOG(libfaceDEBUG) << "Releasing Face Data";
	if(face) {
		cvReleaseImage(&face);
	}
}

Face::Face(int x1, int y1, int x2, int y2, int id, IplImage* face) : d(new FacePriv(x1, y1, x2, y2, id, face)) {}

Face::Face(const Face& that) : d(that.d ? new FacePriv(*that.d) : 0) {
    if(!d) {
        LOG(libfaceERROR) << "Face::Face(const Face& that) : d points to NULL.";
    }
}

Face& Face::operator = (const Face& that) {
    if(this == &that) {
        return *this;
    }
    if( (that.d == 0) || (d == 0) ) {
        LOG(libfaceERROR) << "Face::operator = (const Face& that) : d or that.d points to NULL.";
    } else {
        *d = *that.d;
    }
    return *this;
}

Face::~Face() {
	delete d;
}

void Face::setX1(int x1) {
	d->x1    = x1;
	d->width = d->x2 - d->x1;
}

void Face::setX2(int x2) {
	d->x2    = x2;
	d->width = d->x2 - d->x1;
}

void Face::setY1(int y1) {
	d->y1     = y1;
	d->height = d->y2 - d->y1;
}

void Face::setY2(int y2) {
	d->y2     = y2;
	d->height = d->y2 - d->y1;
}

void Face::setId(int id) {
	d->id = id;
}

void Face::setName(string tag){
    d->tagName = tag;
}

void Face::setFace(IplImage* face) {
    // if another image was already set as d->face, release it
    if(d->face)
        cvReleaseImage(&d->face);
	d->face = face;
}

IplImage* Face::getFace() const {
    return d->face;
}

int Face::getHeight() const {
	return d->height;
}

int Face::getWidth() const {
	return d->width;
}

int Face::getX1() const {
	return d->x1;
}

int Face::getX2() const {
	return d->x2;
}

int Face::getY1() const {
	return d->y1;
}

int Face::getY2() const {
	return d->y2;
}

int Face::getId() const {
	return d->id;
}

string Face::getName() const{
    return d->tagName;
}

} // namespace libface
