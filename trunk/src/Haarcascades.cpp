/** ===========================================================
 * @file Haarcascades.cpp
 *
 * This file is a part of libface project
 * <a href="http://libface.sourceforge.net">http://libface.sourceforge.net</a>
 *
 * @date    2010-03-14
 * @brief   Haar cascades parser
 * @section DESCRIPTION
 *
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

// own header
#include "Haarcascades.h"

// TODO: LOTS of exception handling

using namespace std;

namespace libface
{

CascadeStruct::CascadeStruct() : name(), haarcasc(0) {};

CascadeStruct::CascadeStruct(const string& argName, const string& argFile) : name(argName), haarcasc(0) {
    // TODO If name is always the filename, the c'tor could be simplified to only take on argument.
    // TODO Consider checking if argFile actually exists?
    haarcasc = (CvHaarClassifierCascade*) cvLoad(argFile.c_str(), 0, 0, 0);
};

CascadeStruct::CascadeStruct(const CascadeStruct& that) : name(that.name), haarcasc(0) {
    if(that.haarcasc) {
        haarcasc = (CvHaarClassifierCascade*) cvClone(that.haarcasc);
    }
};

CascadeStruct& CascadeStruct::operator = (const CascadeStruct& that) {
    if(this == &that) {
        return *this;
    }
    name = that.name;
    if(that.haarcasc) {
        haarcasc = (CvHaarClassifierCascade*) cvClone(that.haarcasc);
    }
    return *this;
}

CascadeStruct::~CascadeStruct() {
    if(haarcasc) {
        cvReleaseHaarClassifierCascade(&haarcasc);
    }
}

class Haarcascades::HaarcascadesPriv
{

public:

    HaarcascadesPriv() : cascadePath(), cascades(), weights(), size(0) {}
    HaarcascadesPriv(const string& path) : cascadePath(path), cascades(), weights(), size(0) {}

    // Custom copy constructors, destructor, etc. are not required as long there are no pointer data members.

    string cascadePath;
    vector<Cascade> cascades;
    vector<int> weights;
    int size;
};

Haarcascades::Haarcascades(const string& path) : d(new HaarcascadesPriv(path)) {}

Haarcascades::Haarcascades(const Haarcascades& that) : d(new HaarcascadesPriv(*that.d)) {}

/*
// This operator cannot be used because d is const, but overwriting the auto generated operator might be a good idea.
Haarcascades& Haarcascades::operator = (const Haarcascades& that) {
    if(this == &that) {
        return *this;
    }
    d = new HaarcascadesPriv(*that.d);
}
*/

Haarcascades::~Haarcascades()
{
    // deleting the pointer calls the d'tor of d, which calls the d'tors of all members of the class HaarcascadesPriv
    delete d;
}

void Haarcascades::addCascade(const Cascade& newCascade, const int& newWeight)
{
    if (this->hasCascade(newCascade.name)) {
        return;
    }

    d->cascades.push_back(newCascade);
    d->weights.push_back(newWeight);
    d->size++;
}

void Haarcascades::addCascade(const string& name, const int& newWeight)
{
    if (this->hasCascade(name)) {
        return;
    }

    Cascade newCascade(name, (d->cascadePath + string("/") + name));
    this->addCascade(newCascade, newWeight);
}

bool Haarcascades::hasCascade(const string& name) const
{
    for (int i = 0; i < d->size-1; ++i) {
        if (name == d->cascades.at(i).name) {
            return true;
        }
    }
    return false;
}

void Haarcascades::removeCascade(const string& name)
{
    int i;
    for (i = 0; i < d->size-1; ++i) {
        if (name == d->cascades.at(i).name) {
            break;
        }
    }

    d->cascades.erase(d->cascades.begin() + i);
    d->weights.erase(d->weights.begin() + i);
    d->size--;
}

void Haarcascades::removeCascade(int index)
{
    d->cascades.erase(d->cascades.begin() + index);
    d->weights.erase(d->weights.begin() + index);
    d->size--;
}

int Haarcascades::getWeight(const string& name) const
{
    for (int i = 0; i < d->size-1; ++i) {
        if (name == d->cascades.at(i).name) {
            return d->weights.at(i);
        }
    }
    return -1;	// No such name found, return -1
}

int Haarcascades::getWeight(int index) const
{
    return d->weights.at(index);
}

void Haarcascades::setWeight(const string& name, int weight)
{
    int i;
    for (i = 0; i < d->size-1; ++i)
    {
        if (name == d->cascades.at(i).name)
            break;
    }

    d->weights.at(i) = weight;
}

void Haarcascades::setWeight(int index, int weight)
{
    d->weights.at(index) = weight;
}

const Cascade& Haarcascades::getCascade(const string& name) const
{
    int i;
    for (i = 0; i < d->size-1; ++i)
    {
        if (name == d->cascades.at(i).name)
            break;
    }
    return d->cascades.at(i);
}

const Cascade& Haarcascades::getCascade(int index) const
{
    return d->cascades.at(index);
}

int Haarcascades::getSize() const
{
    return d->size;
}

void Haarcascades::clear()
{
    d->cascades.clear();
    d->weights.clear();
    d->size = 0;
}

} // namespace libface
