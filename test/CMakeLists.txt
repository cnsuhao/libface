#
# This file is part of libface.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
#  Created on: Dec 22, 2010
#  @author Alex Jironkin
#

INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
ADD_EXECUTABLE(testDetection testDetection.cpp)

TARGET_LINK_LIBRARIES(testDetection face ${OpenCV_LIBRARIES})

ADD_TEST(TestDetection testDetection /Users/Aleksey/workspace/test_images/ORL/s1/ 1)