/*
 * testDetection.cpp
 *
 *  Created on: Dec 22, 2010
 *      Author: Aleksey
 */

#include <dirent.h>

#include "../src/LibFace.h";
#include "../src/Face.h";

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

	LibFace* libFace = new LibFace(mode, string("."));




	printf("List Files in %s\n", path);


	DIR *dir;
	struct dirent *ent;
	dir = opendir (path);
	int failed = 0, correct = 0, falsePos = 0;
	if (dir != NULL) {

		/* print all the files and directories within directory */
		while ((ent = readdir (dir)) != NULL) {
			char* filename = ent->d_name;
			if(*filename != '.') {
				printf ("%s\n", filename);

				char tempPath[1024] ;
				strcpy(tempPath, path);
				vector<Face> result = libFace->detectFaces(string(strcat(tempPath,filename)));

				if(result.empty()) {
					failed++;
					printf("No Face Found in %s\n",filename);
				} else if(result.size() > 1) {
					falsePos = falsePos + result.size() - 1;
					correct++;
				} else {
					correct++;
				}

			}
		}
		closedir (dir);
	} else {
		/* could not open directory */
		perror ("");
		return EXIT_FAILURE;
	}


	printf("RESULTS:\n");
	printf("\tCORRECT:\t\t%d\n",correct);
	printf("\tFALSE POSITIVES:\t%d\n",falsePos);
	printf("\tINCORRECT:\t\t%d\n",failed);
	printf("END OF DETECTION TEST\n");

	return EXIT_SUCCESS;


}
