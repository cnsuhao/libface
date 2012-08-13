#ifndef _HMMCORE_H_
#define _HMMCORE_H_

#if defined (__APPLE__)
#include <cv.h>
#else
#include <opencv/cv.h>
#endif

#include <opencv2/legacy/legacy.hpp>

#include <iostream>
#include <string>
#include <map>
using namespace std;

namespace libface
{

class ContEHMM
{
public:
    bool Release();
    ContEHMM();
    virtual ~ContEHMM();

    bool CreateHMM( int* num_states, int* num_mix, int vect_size );
    int GetVectSize() { return m_vectSize; };

    //IppiEHMM* GetIppiEHMM() { return m_hmm; };
    CvEHMM* GetIppiEHMM() { return m_hmm; };

    bool Save( FILE* file );
    bool Load( FILE* file );


protected:
    CvEHMM* m_hmm;
    int m_vectSize;

};

//class FACEAPI PersonImage
//{
//public:

//    PersonImage(int id,string name):m_id(id),m_name(name){};
//    ~PersonImage();

//    void setID(int id){m_id = id;}
//    void setName(string name){m_name = name;}
//    void addImage(int first, IplImage* second){
//        m_face_images.insert(pair<int,IplImage*>(first,second));
//    }

//    int getID(){return m_id;}
//    string getName(){return m_name;}
//    multimap<int,IplImage*> getFaceImages(){return m_face_images;}

//private:
//    int m_id;
//    string m_name;
//    multimap<int,IplImage*> m_face_images;
//    ContEHMM m_hmm;

//};

} // namespace libface

#endif /* _FISHERCORE_H_ */
