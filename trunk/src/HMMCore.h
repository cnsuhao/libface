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

    // Constructor
    ContEHMM(){

        m_hmm = NULL;
        m_vectSize = 0;

    }

    // Destructor
    virtual ~ContEHMM(){
        if (m_hmm) cvRelease2DHMM( &m_hmm );
        m_vectSize = 0;
    }


    int GetVectSize() { return m_vectSize; };
    CvEHMM* GetIppiEHMM() { return m_hmm; };

    bool CreateHMM( int* num_states, int* num_mix, int vect_size ){

        if (m_hmm) cvRelease2DHMM( &m_hmm );
        m_hmm = 0;

        m_hmm = cvCreate2DHMM( num_states, num_mix, vect_size );

        m_vectSize = vect_size;
        return true;
    }

    bool Release(){

        if (m_hmm)
        {
            cvRelease2DHMM( &m_hmm );
            m_hmm = 0;
        }

        return TRUE;
    }

    bool Save( FILE* file );
    bool Load( FILE* file );


protected:
    CvEHMM* m_hmm;
    int m_vectSize;

};

bool ContEHMM::Save(FILE* file){

    if (!m_hmm) return false;

    // write topology
    fprintf(file, "%s %d\n", "<NumSuperStates>", m_hmm->num_states );
    fprintf(file, "%s ", "<NumStates>");

    for( int i = 0; i < m_hmm->num_states; i++ )
    {
        fprintf(file, "%d ", m_hmm->u.ehmm[i].num_states );
    }
    fprintf(file, "\n");

    fprintf(file, "%s ", "<NumMixtures>");
    for(int i = 0; i < m_hmm->num_states; i++ )
    {
        CvEHMM* ehmm = &(m_hmm->u.ehmm[i]);

        for( int j = 0; j < ehmm->num_states; j++ )
        {
            fprintf(file, "%d ", ehmm->u.state[j].num_mix );
        }
    }
    fprintf(file, "\n");

    fprintf(file, "%s %d\n", "<VecSize>", m_vectSize);

    //consequently write all hmms
    CvEHMM* hmm = m_hmm;
    for( int i = 0; i < m_hmm->num_states + 1; i++ )
    {
        if (hmm->level == 0 )
            fprintf(file, "%s\n", "<BeginEmbeddedHMM>");
        else
            fprintf(file, "%s\n", "<BeginExternalHMM>");

        fprintf(file, "%s %d\n", "<NumStates>", hmm->num_states);

        if (hmm->level == 0 )
        {
            for ( int j = 0; j < hmm->num_states; j++)
            {
                CvEHMMState* state = &(hmm->u.state[j]);

                fprintf(file, "%s %d\n", "<State>", j);
                fprintf(file, "%s %d\n", "<NumMixes>", state->num_mix);

                float* mu = state->mu;
                float* inv_var = state->inv_var;

                for( int m = 0; m < state->num_mix; m++)
                {
                    fprintf(file, "%s %d %s %f\n", "<Mixture>", m, "<Weight>", state->weight[m] );
                    fprintf(file, "%s\n", "<Mean>");

                    for (int k = 0; k < m_vectSize; k++)
                    {
                        fprintf(file, "%f ", mu[0]);
                        mu++;
                    }

                    fprintf(file, "\n");
                    fprintf(file, "%s\n", "<Inverted_Deviation>");

                    for (int k = 0; k < m_vectSize; k++)
                    {
                        fprintf(file, "%f ", inv_var[0]);
                        inv_var++;
                    }
                    fprintf(file, "\n");

                    fprintf(file, "%s %f\n", "<LogVarVal>", state->log_var_val[m] );


                }
            }
        }

        //write the transition probability matrix
        fprintf(file, "%s\n", "<TransP>" );
        float* prob = hmm->transP;

        for (int j = 0; j < hmm->num_states; j++)
        {
            for (int k = 0; k < hmm->num_states; k++)
            {
                fprintf(file, "%f ", *prob);
                prob++;
            }
            fprintf(file, "\n");
        }

        if( hmm->level == 0 )
            fprintf(file, "%s\n", "<EndEmbeddedHMM>");
        else
            fprintf(file, "%s\n", "<EndExternalHMM>");

        hmm = &(m_hmm->u.ehmm[i]);
    }

    return true;
}

bool ContEHMM::Load(FILE* file)
{
    int num_states[128];
    int num_mix[128];
    char temp_char[128];

    // read topology
    fscanf(file, "%s %d\n", temp_char, num_states);
    fscanf(file, "%s ", temp_char);

    for( int i = 0; i < num_states[0]; i++ )
    {
        fscanf(file, "%d ", num_states + i + 1 );
    }

    fscanf(file, "\n");

    //compute total number of internal states
    int total_states = 0;
    for(int i = 0; i < num_states[0]; i++ )
    {
        total_states += num_states[i+1];
    }

    //read number of mixtures
    fscanf(file, "%s ", temp_char);
    for(int i = 0; i < total_states; i++ )
    {
        fscanf(file, "%d ", &num_mix[i] );
    }
    fscanf(file, "\n");

    fscanf(file, "%s %d\n", temp_char, &m_vectSize);

    m_hmm = cvCreate2DHMM( num_states, num_mix, m_vectSize);

    //create HMM with known parameters
    //!!!    cvCreate2DHMM( &m_hmm, num_states, num_mix, m_vectSize);
    if (!m_hmm ) return false;

    //consequently read all hmms
    CvEHMM* hmm = m_hmm;
    for(int i = 0; i < num_states[0]+1; i++ )
    {
        fscanf(file, "%s\n", temp_char);
        int temp_int;
        fscanf(file, "%s %d\n", temp_char , &temp_int);  assert(temp_int==num_states[i]);

        if ( i!= 0 )
        {
            for (int j = 0; j < num_states[i]; j++)
            {
                CvEHMMState* state = &(hmm->u.state[j]);

                fscanf(file, "%s %d\n", temp_char, &temp_int); assert(temp_int == j);

                fscanf(file, "%s %d\n", temp_char, &temp_int); assert(temp_int == state->num_mix);

                float* mu = state->mu;
                float* inv_var = state->inv_var;

                for( int m = 0; m < state->num_mix; m++)
                {
                    int temp_int;
                    fscanf(file, "%s %d %s %f\n", temp_char, &temp_int, temp_char, &(state->weight[m]) );
                    assert( temp_int == m );
                    fscanf(file, "%s\n", temp_char );

                    for (int k = 0; k < m_vectSize; k++)
                    {
                        fscanf(file, "%f ", mu);
                        mu++;
                    }

                    fscanf(file, "\n");
                    fscanf(file, "%s\n", temp_char);

                    for (int k = 0; k < m_vectSize; k++)
                    {
                        fscanf(file, "%f ", inv_var);
                        inv_var++;
                    }
                    fscanf(file, "\n");

                    fscanf(file, "%s %f\n", temp_char, &(state->log_var_val[m]) );

                }
            }
        }

        //read the transition probability matrix
        fscanf(file, "%s\n", temp_char );
        float* prob = hmm->transP;

        for (int j = 0; j < hmm->num_states; j++)
        {
            for (int k = 0; k < hmm->num_states; k++)
            {
                fscanf(file, "%f ", prob);
                prob++;
            }
            fscanf(file, "\n");
        }

        fscanf( file, "%s\n", temp_char );

        hmm = &(m_hmm->u.ehmm[i]);
    }

//    fclose(file);
    return true;
}


} // namespace libface

#endif /* _FISHERCORE_H_ */
