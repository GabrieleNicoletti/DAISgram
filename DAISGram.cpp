#include <iostream>
#include <string>

#include "dais_exc.h"
#include "tensor.h"
#include "libbmp.h"
#include "DAISGram.h"

using namespace std;

DAISGram::DAISGram(){};
DAISGram::~DAISGram(){};


Tensor create_filter(float* a){ //CREA FILTRI 3*3
    Tensor res(3, 3, 3);
    for (int i = 0; i < res.cols()*res.rows(); ++i)
        res(i/res.cols(),i%res.cols(),0)=res(i/res.cols(),i%res.cols(),1)=res(i/res.cols(),i%res.cols(),2)=a[i];
    return res;
}

/**
 * Load a bitmap from file
 *
 * @param filename String containing the path of the file
 */
void DAISGram::load_image(string filename){
    BmpImg img = BmpImg();

    img.read(filename.c_str());

    const int h = img.get_height();
    const int w = img.get_width();

    data = Tensor(h, w, 3, 0.0);

    for(int i=0;i<img.get_height();i++){
        for(int j=0;j<img.get_width();j++){
            data(i,j,0) = (float) img.red_at(j,i);
            data(i,j,1) = (float) img.green_at(j,i);
            data(i,j,2) = (float) img.blue_at(j,i);
        }
    }

}


/**
 * Save a DAISGram object to a bitmap file.
 *
 * Data is clamped to 0,255 before saving it.
 *
 * @param filename String containing the path where to store the image.
 */
void DAISGram::save_image(string filename){

    data.clamp(0,255);

    BmpImg img = BmpImg(getCols(), getRows());

    img.init(getCols(), getRows());

    for(int i=0;i<getRows();i++){
        for(int j=0;j<getCols();j++){
            img.set_pixel(j,i,(unsigned char) data(i,j,0),(unsigned char) data(i,j,1),(unsigned char) data(i,j,2));
        }
    }

    img.write(filename);

}


int DAISGram::getRows(){
    return data.rows();
}

int DAISGram::getCols(){
    return data.cols();
}

int DAISGram::getDepth() {
    return data.depth();
}

DAISGram DAISGram::brighten(float bright) {
    DAISGram res{};
    res.data = data + bright;

    res.data.clamp(0, 255);

    return res;
}

DAISGram DAISGram::grayscale(){
    DAISGram res;
    res.data=data;
    int sq=res.data.rows()*res.data.cols();
    int cols=getCols();
    for (int i = 0; i < sq; ++i) {
        res.data(i/cols,i%cols,0)=
        res.data(i/cols,i%cols,1)=
        res.data(i/cols,i%cols,2)=
                (res.data(i/cols,i%cols,0)+res.data(i/cols,i%cols,1)+res.data(i/cols,i%cols,2))/3;
    }
    res.data.clamp(0,255);
    return res;
}

DAISGram DAISGram::warhol(){
  DAISGram res;
  int r, c, d;
  r = getRows();
  c = getCols();
  d = getDepth();
  res.data = Tensor(r*2, c*2, d, 0.0);
  for (int i=0; i<r; i++){
    for (int j=0; j<c; j++){
      res.data(i,j,0)=res.data(i,j+c,1)=res.data(i+r,j,0)=res.data(i+r,j+c,2)=data(i,j,0);
      res.data(i,j,1)=res.data(i,j+c,0)=res.data(i+r,j,2)=res.data(i+r,j+c,1)=data(i,j,1);
      res.data(i,j,2)=res.data(i,j+c,2)=res.data(i+r,j,1)=res.data(i+r,j+c,0)=data(i,j,2);
    }
  }
  return res;
}

DAISGram DAISGram::sharpen(){
    DAISGram res;
    Tensor filter;
    float f[9]= {0, -1 , 0,
                 -1, 5, -1,
                 0, -1, 0};
    filter = create_filter(f);

    res.data = data.convolve(filter);

    res.data.clamp(0, 255);

    return res;
}

DAISGram DAISGram::emboss(){
  DAISGram res;
  Tensor filter;
  float f[9]= {-2, -1, 0,
               -1, 1, 1,
               0, 1, 2};
  filter = create_filter(f);
  res.data = data.convolve(filter);
  res.data.clamp(0, 255);
  return res;
}

DAISGram DAISGram::smooth(int h){
  DAISGram res;
  Tensor filter(h, h, 3, 1.0/(h*h));
  res.data = data.convolve(filter);
  return res;
}

DAISGram DAISGram::edge() {
    DAISGram res;
    Tensor filter;
    float f_data[9]= {-1,-1,-1,
                      -1,8,-1,
                      -1,-1,-1};
    filter=create_filter(f_data);
    res=grayscale();
    res.data=res.data.convolve(filter);
    res.data.clamp(0,255);
    return res;
}

DAISGram DAISGram::blend(const DAISGram & rhs, float alpha){
    if(getCols() != rhs.data.cols() || getRows() != rhs.data.rows() || getDepth() != rhs.data.depth()) throw (dimension_mismatch());
    DAISGram res;
    res.data = this->data * alpha + rhs.data * (1 - alpha);

    return res;
}


DAISGram DAISGram::greenscreen(DAISGram & bkg, int rgb[], float threshold[]){
    if(getRows()!=bkg.getRows() || getCols()!=bkg.getCols() || getDepth()!=bkg.getDepth() ) throw (dimension_mismatch());
    else{
        DAISGram res;
        res.data=data;
        for (int j = 0; j < getRows(); ++j) {
            for (int k = 0; k < getCols(); ++k) {
                bool check=true;

                for (int i = 0; i < getDepth(); ++i)
                    if (this->data(j, k, i) < (float)rgb[i] - threshold[i] || this->data(j, k, i) > (float)rgb[i] + threshold[i]) check = false;

                if(check){
                    for (int l = 0; l < getDepth(); ++l)
                        res.data(j,k,l)=bkg.data(j,k,l);
                }
            }
        }
        return res;
    }
}

DAISGram DAISGram::equalize() {
    DAISGram res;
    int L = 256;
    res.data = data;
    int depth = res.getDepth();
    int cols = res.getCols();
    int rows = res.getRows();

    for (int dim = 0; dim < depth; ++dim) {
        Tensor cdf(1, 256, 1);
        Tensor cdf_n(1, 256, 1);
        float min;
        for (int j = 0; j < rows; ++j) {
            for (int k = 0; k < cols; ++k) {
                cdf(0, (int) this->data(j, k, dim), 0) += 1;
            }
        }

        float sum;
        for (int i = 0; i < L; i++) {
            if (i == 0) sum = cdf(0, i, 0);
            cdf(0, i, 0) = sum;
            if (i < 255)sum += cdf(0, i + 1, 0);
        }

        bool find=false;
        int t=0;
        while (!find) {
            if (cdf(0,t,0) != 0) {
                min = cdf(0,t,0);
                find = true;
            }
            ++t;
        }

       for (int i = 0; i < L; ++i)
            if(cdf(0,i,0)!=0) cdf_n(0,i,0) = round(((float)(cdf(0,i,0) - min) / (float)(data.rows()*data.cols() - min)) * (float)(L - 1));


        for (int j = 0; j < rows; ++j) {
            for (int k = 0; k < cols; ++k) {
                res.data(j, k, dim) = cdf_n(0, (int) this->data(j, k, dim), 0);
            }
        }
    }
    return res;
}



/**
 * Generate Random Image
 *
 * Generate a random image from nois
 *
 * @param h height of the image
 * @param w width of the image
 * @param d number of channels
 * @return returns a new DAISGram containing the generated image.
 */
void DAISGram::generate_random(int h, int w, int d){
    data = Tensor(h,w,d,0.0);
    data.init_random(128,50);
    data.rescale(255);
}
