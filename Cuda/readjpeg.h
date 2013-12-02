/* Adapted from Stanford CS315a Homework */

#ifndef __READJPEG__
#define __READJPEG__

extern "C"
{
#include <jpeglib.h>
}

typedef struct frame_struct {
    JSAMPLE *image_buffer;	/* Points to large array of
                                 * R,G,B-order/grayscale data.
                                 *
                                 * Access directly with:
                                 *   image_buffer[num_components*pixel +
                                 *     component]
                                 */
    JSAMPLE **row_pointers;	/* Points to an array of pointers to the
                                 * beginning of each row in the image buffer.
                                 * Use to access the image buffer in a row-wise
                                 * fashion, with:
                                 *   row_pointers[row][num_components*pixel +
                                 *     component]
                                 */
    int image_height;		/* Number of rows in image */
    int image_width;		/* Number of columns in image */
    int num_components;         /* Number of components (usually RGB=3 or
                                 * gray=1)
                                 */
} frame_struct_t;
typedef frame_struct_t *frame_ptr;

frame_ptr read_JPEG_file (char * filename) ;
void destroy_frame(frame_ptr kill_me);
void write_JPEG_file (char * filename, frame_ptr p_info, int quality);

#endif
