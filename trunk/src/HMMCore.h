#ifndef _HMMCORE_H_
#define _HMMCORE_H_

#if defined (__APPLE__)
#include <cv.h>
#else
#include <opencv/cv.h>
#endif

#include <string>
#include <map>
using namespace std;

namespace libface
{

class FACEAPI PersonImage
{
public:

    PersonImage(int id,string name):m_id(id),m_name(name){};
    ~PersonImage();

    void setID(int id){m_id = id;}
    void setName(string name){m_name = name;}
    void addImage(int first, IplImage* second){
        m_face_images.insert(pair<int,IplImage*>(first,second));
    }

    int getID(){return m_id;}
    string getName(){return m_name;}
    multimap<int,IplImage*> getFaceImages(){return m_face_images;}

private:
    int m_id;
    string m_name;
    multimap<int,IplImage*> m_face_images;

};

} // namespace libface

#endif /* _FISHERCORE_H_ */
