/** ===========================================================
 * @file
 *
 * This file is a part of libface project
 * <a href="http://libface.sourceforge.net">http://libface.sourceforge.net</a>
 *
 * @date    2011-02-08
 * @brief   Detection test.
 * @section DESCRIPTION
 *
 * This is a simple example to test libface in detection mode.
 *
 * @author Copyright (C) 2010 by Alex Jironkin
 *         <a href="alexjironkin at gmail dot com">alexjironkin at gmail dot com</a>
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

#include <dirent.h>
#include <stdio.h>

#include "LibFace.h"
#include "Face.h"

using namespace std;
using namespace libface;

int main(int argc, char* argv[]) {

    if(argc < 3) {
        printf("Wrong Number of parameters. Usage:\n\ttestDetection <input_dir> <num_files>");
        return EXIT_FAILURE;
    }


    char* path = argv[1];
    int n = atoi(argv[2]);

    libface::Mode mode = libface::DETECT;

    //Make an instance of LibFace class with appropriate parameters
    LibFace* libFace = new LibFace(mode, string("."));

    printf("List Files in %s\n", path);


    DIR *dir;
    struct dirent *ent;
    dir = opendir (path);
    int failed = 0, correct = 0, falsePos = 0;
    if (dir != NULL) {

        //print all the files and directories within directory.
        while (((ent = readdir (dir)) != NULL) && (n>0)) {
            char* filename = ent->d_name;
            if(*filename != '.') {
                --n;
                printf("%s\n", filename);

                char tempPath[1024] ;
                strcpy(tempPath, path);
                //Do face detection by calling detectFaces function with file path of an image.
                vector<Face*>* result = libFace->detectFaces(string(strcat(tempPath,filename)));

                if(result->empty()) {
                    failed++;
                    printf("No Face Found in %s\n",filename);
                } else if(result->size() > 1) {
                    falsePos = falsePos + result->size() - 1;
                    correct++;
                } else {
                    correct++;
                }

                delete result;

            }
        }
        closedir (dir);
    } else {
        // could not open directory
        perror ("");
        return EXIT_FAILURE;
    }

    delete libFace;

    printf("RESULTS:\n");
    printf("\tCORRECT:\t\t%d\n",correct);
    printf("\tFALSE POSITIVES:\t%d\n",falsePos);
    printf("\tINCORRECT:\t\t%d\n",failed);
    printf("END OF DETECTION TEST\n");

    return EXIT_SUCCESS;


}
