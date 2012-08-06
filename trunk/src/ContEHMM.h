#ifndef _CONTEHMM_H_
#define _CONTEHMM_H_

// OpenCV headers
#if defined (__APPLE__)
#include <cv.h>
#else
#include <opencv/cv.h>
#endif

#include <opencv2/legacy/legacy.hpp>

class CContEHMM  
{
public:
    bool Release();
	CContEHMM();
	virtual ~CContEHMM(); 

    bool CreateHMM( int* num_states, int* num_mix, int vect_size ); 
    int GetVectSize() { return m_vectSize; };
    
    //IppiEHMM* GetIppiEHMM() { return m_hmm; };
    CvEHMM* GetIppiEHMM() { return m_hmm; };
    

protected:
    CvEHMM* m_hmm;    
    int m_vectSize;
    
};

#endif
