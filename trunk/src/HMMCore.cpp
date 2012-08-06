#include "HMMCore.h"
#include <assert.h>

namespace libface
{

ContEHMM::ContEHMM()
{
    m_hmm = NULL;
    m_vectSize = 0;
}

ContEHMM::~ContEHMM()
{
    if (m_hmm) cvRelease2DHMM( &m_hmm );  
    m_vectSize = 0;

}

bool ContEHMM::CreateHMM( int* num_states, int* num_mix, int vect_size )
{
    if (m_hmm) cvRelease2DHMM( &m_hmm );
    m_hmm = 0;

    m_hmm = cvCreate2DHMM( num_states, num_mix, vect_size ); 
    
    m_vectSize = vect_size;
    return true;
}
    
bool ContEHMM::Release()
{
    if (m_hmm)
    {
        cvRelease2DHMM( &m_hmm ); 
        m_hmm = 0;
    }

    return TRUE;
}

}
