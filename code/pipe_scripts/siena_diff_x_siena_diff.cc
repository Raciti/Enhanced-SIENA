// {{{ Copyright etc.

/*  siena_diff - compute brain change using edge motion or segmentation

    Stephen Smith, FMRIB Image Analysis Group

    Copyright (C) 1999-2006 University of Oxford  */

/*  CCOPYRIGHT */

// }}}
// {{{ includes and options


#include <iostream>
#include <string>

#include <unistd.h>
#include <limits.h>
#include <cstdio> //  Include for printf

#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"

using namespace std;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;


/// MIO 
std::string getPathUntilSiena(const std::string& path) {
  // Find the last occurrence of "Siena" in the path.
  size_t pos = path.find("Siena");
  if (pos != std::string::npos) {
      // If "Siena" is found, extract the substring up to and including "Siena".
      return path.substr(0, pos + 5); // Add 5 to include "Siena"
  } else {
      // If "Siena" is not found, return the original path or an empty string.
      return ""; // Or return path; if you want the whole path even without "Siena"
  }
}
// }}}
// }}}
// {{{ usage

void usage()
{
  cout << "\nCode for the pipeline siena_sienadiff\n" <<
    "\nUsage: siena_diff <input1_basename> <input2_basename> [options] [-s segmentation options]\n\n" <<
    "[-d]            debug - generate edge images and don't remove temporary images\n" <<
    "[-2]            don't segment grey+white separately (because there is poor grey-white contrast)\n" <<
    "[-c <corr>]     apply self-calibrating correction factor\n" <<
    "[-g]            enable GPU processing for synthseg\n" <<
    "[-i]            ignore flow in z (may be good if top of brain is missing)\n" <<
    "[-m]            apply <input1_basename>_stdmask to brain edge points\n" <<
    "[-s <options>]  <options> to be passed to segmentation (type \"fast\" to get these)\n\n" << endl;
    //    "[-e]        erode joint mask a lot instead of dilating it slightly (ie find ventricle surface)\n" <<
  exit(1);
}

// }}}
// {{{ main(argc, argv)

#define CORRWIDTH 3
#define SEARCH    4
#define CS (10*(CORRWIDTH+SEARCH))

int main(int argc,char *argv[])
{
  // {{{  vars

  char cwd[PATH_MAX];
  std::string folderPath;
  std::string synthseg_path; // Declare synthseg_path here.
  std::string currentPath;
  if (getcwd(cwd, sizeof(cwd)) != nullptr) {
      currentPath = cwd;
      folderPath = getPathUntilSiena(currentPath);
      if (!folderPath.empty()) {
        // std::cout << "Path until Siena: " << folderPath << std::endl;
        synthseg_path = folderPath + "/code/pipe_scripts/synthseg.py"; //  Use folderPath
        // std::cout << "SynthSeg Path: " << synthseg_path << std::endl;
      } else {
          std::cout << "Siena directory not found in path: " << currentPath << std::endl;
          //  Handle the case where Siena is not found.  Don't use folderPath here, it's empty
          synthseg_path = currentPath + "/code/pipe_scripts/synthseg.py"; //  use currentPath instead
          std::cout << "SynthSeg Path: " << synthseg_path << std::endl;
      }
    } else {
        std::cerr << "Error getting current working directory" << std::endl;
        return 1;
    }
  // std::cout << "Folder path: " << folderPath << std::endl;
  // std::cout << "Sinthseg path: " << synthseg_path << std::endl;

  char   thestring[10000], segoptions[10000], fsldir[10000];
  int    x_size, y_size, z_size, size, x, y, z, i, count,
    seg2=0, ignore_z=0, ignore_top_slices=0, //erode_mask=0,
    ignore_bottom_slices=0, debug=0, flow_output=1, edge_masking=0, use_gpu=0;
  float  tmpf, calib=1.0, ex, ey, ez;
  ColumnVector arrA(2*CS+1), arrB(2*CS+1), arr1(2*CS+1), arr2(2*CS+1);
  double total, voxel_volume, voxel_area;

  // }}}

  // {{{  process arguments

  if (argc<3)
    usage();

  string argv1(argv[1]), argv2(argv[2]);

  sprintf(fsldir,"%s",getenv("FSLDIR"));

  for (i = 3; i < argc; i++)
    {
      if (!strcmp(argv[i], "-i"))
        ignore_z=1;
      //  else if (!strcmp(argv[i], "-e"))
      //  erode_mask=1;
      else if (!strcmp(argv[i], "-d"))
        debug=1;
      else if (!strcmp(argv[i], "-g"))
        use_gpu=1;
      else if (!strcmp(argv[i], "-2"))
        seg2=1;
      else if (!strcmp(argv[i], "-c"))
        // {{{  apply self-calibrating factor
        {
          i++;

          if (argc<i+1)
            {
              printf("Error: no factor given following -c\n");
              usage();
            }

          calib=atof(argv[i]);
        }

      // }}}
      else if (!strcmp(argv[i], "-m"))
        edge_masking=1;
      else if (!strcmp(argv[i], "-t"))
        // {{{  ignore n slices at top
        {
          i++;

          if (argc<i+1)
            {
              printf("Error: no number of slices given following -t\n");
              usage();
            }

          ignore_top_slices=atoi(argv[i]);
        }

      // }}}
      else if (!strcmp(argv[i], "-b"))
        // {{{  ignore n slices at bottom
        {
          i++;

          if (argc<i+1)
            {
              printf("Error: no number of slices given following -b\n");
              usage();
            }

          ignore_bottom_slices=atoi(argv[i]);
        }

      // }}}
      else if (!strcmp(argv[i], "-s"))
        // {{{  segmentation options
        {
          i++;

          segoptions[0]=0;

          while(i<argc)
            {
              strcat(segoptions,argv[i]);
              strcat(segoptions," ");
              i++;
            }
        }

      // }}}
      else
        usage();
    }

  // }}}
  // {{{  transform images and masks

  sprintf(thestring,"%s/bin/flirt -o %s_halfwayto_%s -applyisoxfm 1 -paddingsize 0 -init %s_halfwayto_%s.mat -ref %s -in %s",
          fsldir,argv[1],argv[2],argv[1],argv[2],argv[1],argv[1]);
  printf("%s\n",thestring); system(thestring);

  sprintf(thestring,"%s/bin/flirt -o %s_halfwayto_%s -applyisoxfm 1 -paddingsize 0 -init %s_halfwayto_%s.mat -ref %s -in %s",
          fsldir,argv[2],argv[1],argv[2],argv[1],argv[1],argv[2]);
  printf("%s\n",thestring); system(thestring);

  sprintf(thestring,"%s/bin/flirt -o %s_halfwayto_%s_mask -applyisoxfm 1 -paddingsize 0 -init %s_halfwayto_%s.mat -ref %s -in %s_brain_mask",
          fsldir,argv[1],argv[2],argv[1],argv[2],argv[1],argv[1]);
  printf("%s\n",thestring); system(thestring);

  sprintf(thestring,"%s/bin/flirt -o %s_halfwayto_%s_mask -applyisoxfm 1 -paddingsize 0 -init %s_halfwayto_%s.mat -ref %s -in %s_brain_mask",
          fsldir,argv[2],argv[1],argv[2],argv[1],argv[1],argv[2]);
  printf("%s\n",thestring); system(thestring);

  if (edge_masking)
    {
      sprintf(thestring,"%s/bin/flirt -o %s_halfwayto_%s_valid_mask -applyisoxfm 1 -paddingsize 0 -init %s_halfwayto_%s.mat -ref %s -in %s_valid_mask_with_%s",
              fsldir,argv[1],argv[2],argv[1],argv[2],argv[1],argv[1],argv[2]);
      printf("%s\n",thestring); system(thestring);
    }

  // }}}
  // {{{  dilate masks, read transformed images and masks, and combine to jointly-masked transformed images

  cout << "reading and combining transformed masks" << endl;
  volume<float> mask;
  read_volume(mask,argv1+"_halfwayto_"+argv2+"_mask");

  // setup header sizes etc.
  x_size=mask.xsize();
  y_size=mask.ysize();
  z_size=mask.zsize();
  size=x_size*y_size*z_size;
  voxel_volume = abs( mask.xdim() * mask.ydim() * mask.zdim() );
  voxel_area = pow(voxel_volume,((double)0.6666667));
  cout << "final image dimensions = " << x_size << " " << y_size << " " << z_size << ", voxel volume = " << voxel_volume << "mm^3, voxel area = " << voxel_area << "mm^2" << endl;

  // read mask 2 and combine with mask 1
  volume<float> mask2;
  read_volume(mask2,argv2+"_halfwayto_"+argv1+"_mask");
  mask=mask+mask2;
  mask2.destroy();
  mask.binarise(0.5);

  cout << "dilating/eroding combined mask" << endl;
  // if (erode_mask)
  //   {
  //     volume<float>kernel=spherical_kernel(17,mask.xdim(),mask.ydim(),mask.zdim());
  //     mask=morphfilter(mask,kernel,"erodeS");
  //   }
  //  else
  {
    volume<float>kernel=box_kernel(3,3,3);
    mask=morphfilter(mask,kernel,"dilate");
  }

  cout << "reading transformed images and applying mask" << endl;
  volume<float> in1;
  read_volume(in1,argv1+"_halfwayto_"+argv2);
  in1 = (in1-in1.min()) * mask;
  mask.destroy();

  // }}}
  // {{{  do segmentation on image 1

  /*FILE *tmpfp;*/
  /*  sprintf(thestring,"%s_halfwayto_%s_brain_seg.hdr",argv[1],argv[2]);*/
  /*  if((tmpfp=fopen(thestring,"rb"))==NULL)*/ /* test for existing segmentation output */

  if(1) // always done unless the above uncommented and used instead of this test
    {
      if(debug)
      {  cout << "\n--- DEBUGGING VARIABLES ---" << endl;
        cout << "synthseg_path: " << synthseg_path << endl;
        cout << "argv[1]: " << argv[1] << endl;
        cout << "argv[2]: " << argv[2] << endl;
        cout << "currentPath: " << currentPath << endl;
        cout << "use_gpu: " << use_gpu << endl;
        cout << "---------------------------\n" << endl;
      }
      // std::cout << "Folder path: " << folderPath << std::endl;
      // std::cout << "Sinthseg path: " << synthseg_path << std::endl;
      char segtype[100];
      if (seg2) sprintf(segtype,"-n 2"); else segtype[0]=0;
      cout << "saving image 1 to disk prior to segmentation" << endl;
      save_volume(in1,argv1+"_halfwayto_"+argv2+"_brain");
      in1.destroy();
      // sprintf(thestring,"%s/bin/fast %s %s %s_halfwayto_%s_brain > %s_halfwayto_%s_brain.vol 2>&1",
      //         fsldir,segtype,segoptions,argv[1],argv[2],argv[1],argv[2]);
      sprintf(thestring,"python %s --input %s_halfwayto_%s_brain.nii.gz --output_path %s --gpu %d > %s_halfwayto_%s_brain.vol 2>&1",
              synthseg_path.c_str(),argv[1],argv[2],currentPath.c_str(),use_gpu,argv[1],argv[2]);
      cout << thestring << endl;
      system(thestring);
    }
  else
    {
      cout << "using previously carried out segmentation" << endl;
      in1.destroy();
    }

  // }}}
  // {{{  read segmentation output into edges1 and simplify; reread in1 and in2

  printf("finding brain edges\n");

  volume<float> seg1;
  read_volume(seg1,argv1+"_halfwayto_"+argv2+"_brain_seg");
  // read_volume(seg1,argv1+"_halfwayto_"+argv2+"_brain_pveseg");
  seg1.binarise(1.5);

  volume<float> m1;
  if (edge_masking)
    read_volume(m1,argv1+"_halfwayto_"+argv2+"_valid_mask");

  read_volume(in1,argv1+"_halfwayto_"+argv2);
  in1.setinterpolationmethod(trilinear);

  volume<float> in2;
  read_volume(in2,argv2+"_halfwayto_"+argv1);
  in2.setinterpolationmethod(trilinear);

  // }}}
  // {{{  find segmentation-based edges in image 1 and flow

  printf("finding flow\n");

  volume<float> flow=in1;
  flow=0;

  // requested addition to output edge points for viena
  volume<float> edgepts=in1;
  edgepts=0;

  count=0;
  total=0;

  ignore_bottom_slices=max(1,ignore_bottom_slices);
  ignore_top_slices=max(1,ignore_top_slices);

  for (z=ignore_bottom_slices; z<z_size-ignore_top_slices; z++)
    for (y=1; y<y_size-1; y++)
      for (x=1; x<x_size-1; x++)
        {
          if ( (seg1(x,y,z)>0.5) &&            /* not background or CSF */
               ( (seg1(x+1,y,z)<0.5) || (seg1(x-1,y,z)<0.5) ||
                 (seg1(x,y+1,z)<0.5) || (seg1(x,y-1,z)<0.5) ||
                 (seg1(x,y,z+1)<0.5) || (seg1(x,y,z-1)<0.5) ) &&
               ( ( ! edge_masking ) || ( m1(x,y,z)>0 ) ) )
            {
              int pos, neg, r, rr, rrr, d, X, Y, Z;
              float ss, maxss, segvalpos=0, segvalneg=0;

              // {{{  find local gradient and derive unit normal

              ex = ( 10*(in1(x+1,y,z)-in1(x-1,y,z)) +
                     5*(in1(x+1,y+1,z)+in1(x+1,y-1,z)+in1(x+1,y,z+1)+in1(x+1,y,z-1)-
                        in1(x-1,y+1,z)-in1(x-1,y-1,z)-in1(x-1,y,z+1)-in1(x-1,y,z-1)) +
                     2*(in1(x+1,y+1,z+1)+in1(x+1,y-1,z+1)+in1(x+1,y+1,z-1)+in1(x+1,y-1,z-1)-
                        in1(x-1,y+1,z+1)-in1(x-1,y-1,z+1)-in1(x-1,y+1,z-1)-in1(x-1,y-1,z-1)) ) / 38;
              ey = ( 10*(in1(x,y+1,z)-in1(x,y-1,z)) +
                     5*(in1(x+1,y+1,z)+in1(x-1,y+1,z)+in1(x,y+1,z+1)+in1(x,y+1,z-1)-
                        in1(x+1,y-1,z)-in1(x-1,y-1,z)-in1(x,y-1,z+1)-in1(x,y-1,z-1)) +
                     2*(in1(x+1,y+1,z+1)+in1(x-1,y+1,z+1)+in1(x+1,y+1,z-1)+in1(x-1,y+1,z-1)-
                        in1(x+1,y-1,z+1)-in1(x-1,y-1,z+1)-in1(x+1,y-1,z-1)-in1(x-1,y-1,z-1)) ) / 38;
              ez = ( 10*(in1(x,y,z+1)-in1(x,y,z-1)) +
                     5*(in1(x,y+1,z+1)+in1(x,y-1,z+1)+in1(x+1,y,z+1)+in1(x-1,y,z+1)-
                        in1(x,y+1,z-1)-in1(x,y-1,z-1)-in1(x+1,y,z-1)-in1(x-1,y,z-1)) +
                     2*(in1(x+1,y+1,z+1)+in1(x+1,y-1,z+1)+in1(x-1,y+1,z+1)+in1(x-1,y-1,z+1)-
                        in1(x+1,y+1,z-1)-in1(x+1,y-1,z-1)-in1(x-1,y+1,z-1)-in1(x-1,y-1,z-1)) ) / 38;

              tmpf = sqrt(ex*ex+ey*ey+ez*ez);

              if (tmpf>0)
                {
                  ex/=(double)tmpf;
                  ey/=(double)tmpf;
                  ez/=(double)tmpf;
                }

              // }}}

              if ( (!ignore_z) ||
                   ( (abs(ez)<abs(ex)) && (abs(ez)<abs(ey)) ) )
                {
                  // {{{  fill 1D arrays and differentiate TLI

                  arrA=0; arrB=0; arr1=0; arr2=0;

                  /*flow(x,y,z) = 1;*/ /* DEBUG colour edge point */

                  /*if ((x==53)&&(y==61)&&(z==78)) {*/  /* DEBUG */

                  /*  printf("normal=(%f %f %f) ",ex,ey,ez);*/ /* DEBUG */

                  arrA(CS)=in1(x,y,z);
                  arrB(CS)=in2(x,y,z);

                  /*flow(x,y,z) = 3;*/ /* DEBUG colour central point */

                  pos=0;
                  d=1; X=MISCMATHS::round(x+d*ex); Y=MISCMATHS::round(y+d*ey); Z=MISCMATHS::round(z+d*ez);
                  if ( (X>0) && (X<x_size-1) && (Y>0) && (Y<y_size-1) && (Z>0) && (Z<z_size-1) )
                    {
                      arrA(CS+1)=in1.interpolate(x+d*ex,y+d*ey,z+d*ez);
                      arrB(CS+1)=in2.interpolate(x+d*ex,y+d*ey,z+d*ez);
                      pos=-1;
                      segvalpos = seg1(X,Y,Z);
                      for(d=2;d<=CORRWIDTH+SEARCH+1;d++)
                        {
                          X=MISCMATHS::round(x+d*ex); Y=MISCMATHS::round(y+d*ey); Z=MISCMATHS::round(z+d*ez);
                          if ( (X>0) && (X<x_size-1) && (Y>0) && (Y<y_size-1) && (Z>0) && (Z<z_size-1) )
                            {
                              if ( (pos<0) && (seg1(X,Y,Z)!=segvalpos) )
                                pos=d-1;
                              arrA(CS+d)=in1.interpolate(x+d*ex,y+d*ey,z+d*ez);
                              arrB(CS+d)=in2.interpolate(x+d*ex,y+d*ey,z+d*ez);
                            }
                          else
                            break;
                        }
                      if ( (pos<0) || (pos>CORRWIDTH) )
                        pos=CORRWIDTH;
                      if (pos==d-1)
                        pos=d-2;
                    }

                  // {{{  COMMENT DEBUG draw search space

#ifdef FoldingComment

                  for(d=1;d<=SEARCH+pos;d++)
                    {
                      X=MISCMATHS::round(x+d*ex); Y=MISCMATHS::round(y+d*ey); Z=MISCMATHS::round(z+d*ez);
                      if (d<=pos)
                        flow(X,Y,Z) = 7;
                      else
                        flow(X,Y,Z) = 5;
                    }

#endif

                  // }}}

                  neg=0;
                  d=-1; X=MISCMATHS::round(x+d*ex); Y=MISCMATHS::round(y+d*ey); Z=MISCMATHS::round(z+d*ez);
                  if ( (X>0) && (X<x_size-1) && (Y>0) && (Y<y_size-1) && (Z>0) && (Z<z_size-1) )
                    {
                      arrA(CS-1)=in1.interpolate(x+d*ex,y+d*ey,z+d*ez);
                      arrB(CS-1)=in2.interpolate(x+d*ex,y+d*ey,z+d*ez);
                      neg=1;
                      segvalneg = seg1(X,Y,Z);
                      for(d=-2;d>=-CORRWIDTH-SEARCH-1;d--)
                        {
                          X=MISCMATHS::round(x+d*ex); Y=MISCMATHS::round(y+d*ey); Z=MISCMATHS::round(z+d*ez);
                          if ( (X>0) && (X<x_size-1) && (Y>0) && (Y<y_size-1) && (Z>0) && (Z<z_size-1) )
                            {
                              if ( (neg>0) && (seg1(X,Y,Z)!=segvalneg) )
                                neg=d+1;
                              arrA(CS+d)=in1.interpolate(x+d*ex,y+d*ey,z+d*ez);
                              arrB(CS+d)=in2.interpolate(x+d*ex,y+d*ey,z+d*ez);
                            }
                          else
                            break;
                        }
                      if ( (neg>0) || (neg<-CORRWIDTH) )
                        neg=-CORRWIDTH;
                      if (neg==d+1)
                        neg=d+2;
                    }

                  // {{{  COMMENT DEBUG draw search space

#ifdef FoldingComment

                  for(d=-1;d>=-SEARCH+neg;d--)
                    {
                      X=MISCMATHS::round(x+d*ex); Y=MISCMATHS::round(y+d*ey); Z=MISCMATHS::round(z+d*ez);
                      if (d>=neg)
                        flow(X,Y,Z) = 7;
                      else
                        flow(X,Y,Z) = 5;
                    }

#endif

                  // }}}

                  /*printf("<%d %d %d %d>  ",neg,pos,(int)segvalneg,(int)segvalpos);*/  /* DEBUG*/

                  for(d=-SEARCH-CORRWIDTH-1;d<=SEARCH+CORRWIDTH+1;d++)
                    {
                      float denom = max(1,pos-neg);
                      arr1(CS+d)=exp(-0.5*pow((2.0*d-neg-pos)/denom,4.0)) * (arrA(CS+d+1)-arrA(CS+d-1));
                      arr2(CS+d)=exp(-0.5*pow((2.0*d-neg-pos)/denom,4.0)) * (arrB(CS+d+1)-arrB(CS+d-1));
                    }

                  // }}}
                  // {{{  find position of maximum correlation

                  for(r=-SEARCH, maxss=0, rrr=0; r<=SEARCH; r++)
                    {
                      for(rr=neg, ss=0; rr<=pos; rr++)
                        ss+=arr1(CS+rr)*arr2(CS+rr+r);

                      arrA(CS+r)=ss;

                      /*  printf("[%d %.2f] ",r,ss);*/ /* DEBUG */

                      if ( (ss>maxss) && (r>-SEARCH) && (r<SEARCH) )
                        {
                          maxss=ss;
                          rrr=r;
                        }
                    }

                  /* now find this max to sub-voxel accuracy */
                  tmpf = arrA(CS+rrr+1) + arrA(CS+rrr-1) - 2*arrA(CS+rrr);
                  if (tmpf!=0)
                    tmpf = 0.5 * (arrA(CS+rrr-1)-arrA(CS+rrr+1)) / tmpf;

                  if ( (tmpf<-0.5) || (tmpf>0.5) ) /* protect against sub-voxel fit not making sense */
                    tmpf=0;
                  else
                    tmpf+=rrr;

                  tmpf = (segvalneg-segvalpos)*tmpf; /* use segmentation info to get directionality */

                  /*printf(" tmpf=%f\n",tmpf);*/ /* DEBUG */

                  flow(x,y,z) = tmpf; /* turn off if DEBUGging */

                  // set edge points
                  edgepts(x,y,z) = 1;

                  total += tmpf;
                  count ++;

                  /*}*/ /* DEBUG */

                  // }}}
                } // end: !ignore_z ...
            } // end: valid voxel
        } // end: voxel loop

  // }}}
  // {{{  final outputs

  // save edge points
  save_volume(edgepts,argv1+"_to_"+argv2+"_edgepoints");

  if (flow_output)
    save_volume(flow,argv1+"_to_"+argv2+"_flow");

  cout << "AREA  " << count*voxel_area << " mm^2" << endl;
  cout << "VOLC  " << total*voxel_volume << " mm^3" << endl;
  cout << "RATIO " << (total*voxel_volume) / (count*voxel_area) << " mm" << endl;    /* mean perpendicular edge motion; l in the equations */
  cout << "PBVC  " << (calib*30*total*voxel_volume) / (count*voxel_area) << " %" << endl;

  // }}}
  return 0;
}

// }}}
