#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "readjpeg.h"
/* Adapted from Stanford CS315a Homework */

static frame_ptr allocate_frame(int height, int width, int num_components) {
  int row_stride;		/* physical row width in output buffer */
  int i;
  frame_ptr p_info;		/* Output frame information */
  
  /* JSAMPLEs per row in output buffer */
  row_stride = width * num_components;
  
  /* Basic struct and information */
  
  p_info = (frame_ptr)malloc(sizeof(frame_struct_t));
  
  p_info->image_height = height;
  p_info->image_width = width;
  p_info->num_components = num_components;
  
  /* Image array and pointers to rows */
  p_info->row_pointers = (JSAMPLE**)malloc(sizeof(JSAMPLE *) * height);
  
  
  
  p_info->image_buffer = (JSAMPLE*)malloc(sizeof(JSAMPLE) * row_stride * 
					  height);
  
  
  for (i=0; i < height; i++)
    p_info->row_pointers[i] = & (p_info->image_buffer[i * row_stride]);
  
  /* And send it back! */
  return p_info;
}

frame_ptr read_JPEG_file (char * filename) 
{
  /* This struct contains the JPEG decompression parameters and pointers to
   * working space (which is allocated as needed by the JPEG library).
   */
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  FILE * infile;		/* source file */
  frame_ptr p_info;		/* Output frame information */
  
  
  /* Step 1: allocate and initialize JPEG decompression object */
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  
  /* Step 2: open & specify data source (eg, a file) */
  if ((infile = fopen(filename, "rb")) == NULL) {
    fprintf(stderr, "ERROR: Can't open input file %s\n", filename);
    exit(1);
  }
  jpeg_stdio_src(&cinfo, infile);
  
  /* Step 3: read file parameters with jpeg_read_header() */
  (void) jpeg_read_header(&cinfo, TRUE);
  
  /* Step 4: use default parameters for decompression */
  
  /* Step 5: Start decompressor */
  (void) jpeg_start_decompress(&cinfo);
  
  /* Step X: Create a frame struct & buffers and fill in the blanks */
  fprintf(stderr, "  Opened %s: height = %d, width = %d, c = %d\n",
	  filename, cinfo.output_height, cinfo.output_width, cinfo.output_components);
  p_info = allocate_frame(cinfo.output_height, cinfo.output_width, cinfo.output_components);
  
  /* Step 6: while (scan lines remain to be read) */
  /*           jpeg_read_scanlines(...); */
  while (cinfo.output_scanline < cinfo.output_height) {
    (void) jpeg_read_scanlines(&cinfo, &(p_info->row_pointers[cinfo.output_scanline]), 1);
  }
  
  /* Step 7: Finish decompression */
  (void) jpeg_finish_decompress(&cinfo);
  
  /* Step 8: Release JPEG decompression object & file */
  jpeg_destroy_decompress(&cinfo);
  fclose(infile);
  
  /* At this point you may want to check to see whether any corrupt-data
   * warnings occurred (test whether jerr.pub.num_warnings is nonzero).
   */

  /* And we're done! */
  return p_info;
}


void destroy_frame(frame_ptr kill_me) 
{
  free(kill_me->image_buffer);
  free(kill_me->row_pointers);
  free(kill_me);
}


void write_JPEG_file (char * filename, frame_ptr p_info, int quality) {
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  FILE * outfile;		/* target file */
  
  /* Step 1: allocate and initialize JPEG compression object */
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  
  /* Step 2: specify data destination (eg, a file) */
  /* Note: steps 2 and 3 can be done in either order. */
  
  if ((outfile = fopen(filename, "wb")) == NULL) {
    fprintf(stderr, "ERROR: Can't open output file %s\n", filename);
    exit(1);
  }
  jpeg_stdio_dest(&cinfo, outfile);
  
  /* Step 3: set parameters for compression */
  
  /* Set basic picture parameters (not optional) */
  cinfo.image_width = p_info->image_width; 	/* image width and height, in pixels */
  cinfo.image_height = p_info->image_height;
  cinfo.input_components = p_info->num_components; /* # of color components per pixel */
  if (p_info->num_components == 3)
    cinfo.in_color_space = JCS_RGB; 	/* colorspace of input image */
  else if (p_info->num_components == 1)
    cinfo.in_color_space = JCS_GRAYSCALE;
  else {
    fprintf(stderr, "ERROR: Non-standard colorspace for compressing!\n");
    exit(1);
  } 
  /* Fill in the defaults for everything else, then override quality */
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);
  
  /* Step 4: Start compressor */
  jpeg_start_compress(&cinfo, TRUE);
  
  /* Step 5: while (scan lines remain to be written) */
  /*           jpeg_write_scanlines(...); */
  while (cinfo.next_scanline < cinfo.image_height) {
    (void) jpeg_write_scanlines(&cinfo, &(p_info->row_pointers[cinfo.next_scanline]), 1);
  }
  
  /* Step 6: Finish compression & close output */
  
  jpeg_finish_compress(&cinfo);
  fclose(outfile);
  
  /* Step 7: release JPEG compression object */
  jpeg_destroy_compress(&cinfo);
}
