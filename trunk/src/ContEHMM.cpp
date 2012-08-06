#include "ContEHMM.h"
#include <assert.h>


CContEHMM::CContEHMM()
{
    m_hmm = NULL;
    m_vectSize = 0;
}

CContEHMM::~CContEHMM()
{
    if (m_hmm) cvRelease2DHMM( &m_hmm );  
    m_vectSize = 0;

}

bool CContEHMM::CreateHMM( int* num_states, int* num_mix, int vect_size )
{
    if (m_hmm) cvRelease2DHMM( &m_hmm );
    m_hmm = 0;

    m_hmm = cvCreate2DHMM( num_states, num_mix, vect_size ); 
    
    m_vectSize = vect_size;
    return true;
}
    
bool CContEHMM::Release()
{
    if (m_hmm)
    {
        cvRelease2DHMM( &m_hmm ); 
        m_hmm = 0;
    }

    return TRUE;
}
