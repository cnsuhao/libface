/** ===========================================================
 * @file Log.h
 *
 * This file is a part of libface project
 * <a href="http://libface.sourceforge.net">http://libface.sourceforge.net</a>
 *
 * @date    2010-02-18
 * @brief   Logging class.
 * @section DESCRIPTION
 * 	This class has been adopted from the Dr. Dobb's <a href="http://www.drdobbs.com/cpp/201804215">article</a>.
 *
 * @author Copyright (C) 2011 adopted  by Alex Jironkin
 *         <a href="alexjironkin at gmail dot com">alexjironkin at gmail dot com</a>
 * @author Copyright (C) 2007 written by Petru Marginean
 *
 *  Created on: Feb 16, 2011
 *      Author: Alex Jironkin
 */

#ifndef LOG_H_
#define LOG_H_

#include "LibFaceConfig.h"

#include <sstream>
#include <string>
#include <stdio.h>

namespace libface {

//Simple definition for instansiating Log only when level is below the reporting level, as set during config.
#define LOG(level) \
        if (level > Log::ReportingLevel()) ; \
        else Log().Get(level)

inline std::string NowTime();

enum TLogLevel {libfaceERROR, libfaceWARNING, libfaceINFO, libfaceDEBUG};

class Log {
public:
	Log();
	virtual ~Log();
	std::ostringstream& Get(TLogLevel level = libfaceINFO);
public:
	static TLogLevel& ReportingLevel();
	static std::string ToString(TLogLevel level);
	static TLogLevel FromString(const std::string& level);
protected:
	std::ostringstream os;
private:
	Log(const Log&);
	Log& operator =(const Log&);
};

inline Log::Log() {
}

/**
 * Get the string stream to write log information to.
 *
 * @param level A TLogLevel logging level.
 *
 * @return Returnd std::ostringstream to write log information to.
 */
inline std::ostringstream& Log::Get(TLogLevel level) {
	os << "- " << NowTime();
	os << " " << ToString(level) << ": ";
	os << std::string(level > libfaceDEBUG ? level - libfaceDEBUG : 0, '\t');
	return os;
}

/**
 * log deconstructor. Only during the deconstruction the buffer is flushed. For more information read the article.
 */
inline Log::~Log() {
	os << std::endl;
	fprintf(stderr, "%s", os.str().c_str());
	fflush(stderr);
}

/**
 * Retrieve the reporting level. The level is set during the config time. If the "Debug" was set as
 * CMAKE_BUILD_TYPE, then the level will be set to libfaceDEBUG, otherwise it is set to libfaceINFO. This
 * level is in the LibFaceConfig.h.
 *
 */
inline TLogLevel& Log::ReportingLevel() {
	static TLogLevel reportingLevel = LOG_LEVEL;
	return reportingLevel;
}

/**
 * Convert TLogLevel to a readable string.
 *
 * @param level A TLogLevel logging level.
 *
 * @return Returns a string corresponding to the level.
 */
inline std::string Log::ToString(TLogLevel level) {
	static const char* const buffer[] = {"libfaceERROR", "libfaceWARNING", "libfaceINFO", "libfaceDEBUG"};
	return buffer[level];
}

/**
 * Convert from string to a TLogLevel.
 *
 * @param level A string representing the logging level.
 *
 * @return Returns a TLogLevel representing the level.
 */
inline TLogLevel Log::FromString(const std::string& level) {
	if (level == "libfaceDEBUG")
		return libfaceDEBUG;
	if (level == "libfaceINFO")
		return libfaceINFO;
	if (level == "libfaceWARNING")
		return libfaceWARNING;
	if (level == "libfaceERROR")
		return libfaceERROR;

	Log().Get(libfaceWARNING) << "Unknown logging level '" << level << "'. Using INFO level as default.";
	return libfaceINFO;
}

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)

#include <windows.h>

inline std::string NowTime() {
	const int MAX_LEN = 200;
	char buffer[MAX_LEN];
	if (GetTimeFormatA(LOCALE_USER_DEFAULT, 0, 0,
			"HH':'mm':'ss", buffer, MAX_LEN) == 0)
		return "Error in NowTime()";

	char result[100] = {0};
	static DWORD first = GetTickCount();
	std::sprintf(result, "%s.%03ld", buffer, (long)(GetTickCount() - first) % 1000);
	return result;
}

#else

#include <sys/time.h>

inline std::string NowTime() {
	char buffer[11];
	time_t t;
	time(&t);
	tm r = {0};
	strftime(buffer, sizeof(buffer), "%H:%M:%S", localtime_r(&t, &r));
	struct timeval tv;
	gettimeofday(&tv, 0);
	char result[100] = {0};
	sprintf(result, "%s.%03ld", buffer, (long)tv.tv_usec / 1000);
	return result;
}

#endif //WIN32


} //namespace libface
#endif /* LOG_H_ */
