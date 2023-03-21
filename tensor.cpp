#include <iostream>
#include <string>
#include <random>
#include <math.h>
#include <fstream>

#include "dais_exc.h"
#include "tensor.h"

#define PI 3.141592654
#define FLT_MAX 3.402823466e+38F /* max value */
#define FLT_MIN 1.175494351e-38F /* min positive value */
using namespace std;

Tensor::Tensor() {
    data = nullptr;
    this->c = 0;
    this->r = 0;
    this->d = 0;
}

Tensor::Tensor(int r, int c, int d, float v){
    if(r<0 || c<0 || d<0) throw (unknown_exception());
    this->r = r;
    this->c = c;
    this->d = d;
    if(r*c*d==0) data = nullptr;
    else data = new float[r * c * d];
    for (int i = 0; i < r * c * d; i++)
        data[i] = v;
}

Tensor::~Tensor(){
    delete [] data;
}

float Tensor::operator()(int i, int j, int k) const{
    if(!data)throw (tensor_not_initialized());
    if(i<0 || i>r-1 || j<0 || j>c-1 || k<0 || k>d-1) throw (index_out_of_bound());
    return data[i*c+j+k*(r*c)];
}

float &Tensor::operator()(int i, int j, int k){
    if(!data)throw (tensor_not_initialized());
    if(i<0 || i>r-1 || j<0 || j>c-1 || k<0 || k>d-1) throw (index_out_of_bound());
    return this->data[i*c+j+k*(r*c)];
}

Tensor::Tensor(const Tensor& that){
    if(!that.data)throw (tensor_not_initialized());
    this->r = that.r;
    this->c = that.c;
    this->d = that.d;
    data = new float[r * c * d];
    for (int i = 0; i < r * c * d; ++i)
        data[i] = that.data[i];
}

bool Tensor::operator==(const Tensor& rhs) const{
    if (!data || !rhs.data) throw (tensor_not_initialized());
    if(d!=rhs.d || r!=rhs.r || c!=rhs.c) throw (dimension_mismatch());
    bool res=true;
    int i=0;
    while(i<r*c*d && res){
        float a=data[i];
        float b=rhs.data[i];
        if((abs(a-b)>=EPSILON)) res=false;
        ++i;
    }
    return res;
}

Tensor Tensor::operator-(const Tensor &rhs)const{
    if (!data || !rhs.data) throw (tensor_not_initialized());
    if(d!=rhs.d || r!=rhs.r || c!=rhs.c) throw (dimension_mismatch());
    Tensor res(r,c,d);
    for (int i = 0; i < d*r*c; ++i)
        res.data[i]=data[i]-rhs.data[i];
    return res;
}

Tensor Tensor::operator+(const Tensor &rhs)const{
    if (!data || !rhs.data) throw (tensor_not_initialized());
    if(d!=rhs.d || r!=rhs.r || c!=rhs.c) throw (dimension_mismatch());
    Tensor res(r,c,d);
    for (int i = 0; i < d*r*c; ++i)
        res.data[i]=data[i]+rhs.data[i];
    return res;
}


Tensor Tensor::operator*(const Tensor &rhs)const{
    if (!data || !rhs.data) throw (tensor_not_initialized());
    if(d!=rhs.d || r!=rhs.r || c!=rhs.c) throw (dimension_mismatch());
    Tensor res(r,c,d);
    for (int i = 0; i < d*r*c; ++i)
        res.data[i]=data[i]*rhs.data[i];
    return res;
}

Tensor Tensor::operator/(const Tensor &rhs)const{
    if (!data || !rhs.data) throw (tensor_not_initialized());
    if(d!=rhs.d || r!=rhs.r || c!=rhs.c) throw (dimension_mismatch());
    Tensor res(r,c,d);
    for (int i = 0; i < d*r*c; ++i)
        if(rhs.data[i]==0) throw (unknown_exception());
        else res.data[i]=data[i]/rhs.data[i];
    return res;
}

Tensor Tensor::operator-(const float &rhs)const{
    if (!data) throw (tensor_not_initialized());
    Tensor res(r,c,d);
    for (int i = 0; i < d*r*c; ++i)
        res.data[i]=data[i]-rhs;
    return res;
}

Tensor Tensor::operator+(const float &rhs)const{
    if (!data) throw (tensor_not_initialized());
    Tensor res(r,c,d);
    for (int i = 0; i < d*r*c; ++i)
        res.data[i]=data[i]+rhs;
    return res;
}

Tensor Tensor::operator*(const float &rhs)const{
    if (!data) throw (tensor_not_initialized());
    Tensor res(r,c,d);
    for (int i = 0; i < d*r*c; ++i)
        if(data[i]==0 || rhs==0) res.data[i]=0;
        else res.data[i]=data[i]*rhs;
    return res;
}

Tensor Tensor::operator/(const float &rhs)const{
    if (!data) throw (tensor_not_initialized());
    if (rhs==0) throw (unknown_exception());
    Tensor res(r,c,d);
    for (int i = 0; i < d*r*c; ++i)
         res.data[i]=data[i]/rhs;
    return res;
}

Tensor & Tensor::operator=(const Tensor &other){
        delete[] data;
        d = other.d;
        r = other.r;
        c = other.c;
        data = new float[other.d * other.r * other.c];
        for (int i = 0; i < d * r * c; ++i)
            data[i] = other.data[i];
        return *this;
}


/**
 * Random Initialization
 *
 * Perform a random initialization of the tensor
 *
 * @param mean The mean
 * @param std  Standard deviation
 */
void Tensor::init_random(float mean, float std){
    if(data){

        std::default_random_engine generator;
        std::normal_distribution<float> distribution(mean,std);

        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++){
                for(int k=0;k<d;k++){
                    this->operator()(i,j,k)= distribution(generator);
                }
            }
        }

    }else{
        throw(tensor_not_initialized());
    }
}

void Tensor::init(int r, int c, int d, float v){
    if(r<0 || c<0 || d<0) throw (unknown_exception());
    if(data) delete [] data;
    this->r=r;
    this->c=c;
    this->d=d;
    if(r*c*d==0) data= nullptr;
    else data = new float[r * c * d];
    for (int i = 0; i < r*c*d; ++i)
        data[i]=v;
}

void Tensor::clamp(float low, float high){
    if(!data) throw (tensor_not_initialized());
    if(fmax(low,high)==low && low!=high) throw (unknown_exception());
    for (int i = 0; i < d*r*c; ++i) {
        if(data[i]<low)
            data[i]=low;
        else if (data[i]>high)
            data[i]=high;
    }
}


void Tensor::rescale(float new_max){
    if(!data) throw (tensor_not_initialized());
        float min;
        float max;
        for (long int i = 0; i < r * d * c; ++i) {
            if (i % (r * c) == 0 || i == 0) {
                min = getMin(i / (r * c));
                max = getMax(i / (r * c));
            }
            if((max - min)==0)throw (unknown_exception());
            else data[i] = ((data[i] - min) / (max - min)) * new_max;
        }
}

Tensor Tensor::padding(int pad_h, int pad_w)const{
    if (!data) throw (tensor_not_initialized());
    if(pad_h<0 || pad_w<0) throw (unknown_exception());
    Tensor aux(r+2*pad_h,c+2*pad_w,d);
    int t1=0;

    for (int i = 0; i < aux.d; ++i) {
        for (int j = 0; j <aux.r ; ++j) {
            for (int k = 0; k <aux.c ; ++k) {
                if(j<pad_h || j>aux.r-pad_h-1 || k<pad_w || k>aux.c-pad_w-1) aux(j,k,i)=0.0;
                else{
                    aux(j,k,i)=data[t1];
                    ++t1;
                }
            }
        }
    }
    return aux;
}


Tensor Tensor::subset(unsigned int row_start, unsigned int row_end, unsigned int col_start, unsigned int col_end, unsigned int depth_start, unsigned int depth_end)const{
    if(!data) throw (tensor_not_initialized());
    if(row_start<0 || row_start>row_end || row_end>=r || col_start<0 || col_start>col_end || col_end>=c || depth_start<0 || depth_start>depth_end || depth_end>=d ) throw (index_out_of_bound());
    Tensor aux(row_end-row_start+1, col_end-col_start+1, depth_end-depth_start+1);
    int t1=0;
    for (int i = depth_start; i <depth_end+1 ; ++i) {
        for (int j = row_start; j <row_end+1; ++j) {
            for (int k = col_start; k <col_end+1; ++k) {
                aux.data[t1]=operator()(j,k,i);
                ++t1;
            }
        }
    }
    return aux;

}


Tensor Tensor::concat(const Tensor &rhs, int axis)const{
    if(!data || !rhs.data) throw (tensor_not_initialized());
    if(axis>2 || axis<0) throw (unknown_exception());
    if((axis==0 && (d!=rhs.d || c!=rhs.c)) || (axis==1 && (d!=rhs.d || r!=rhs.r)) || (axis==2 && (c!=rhs.c || r!=rhs.r)) ) throw (concat_wrong_dimension());
    Tensor res;
    int t1=0;
    int t2=0;
    if (axis == 0 ) {
        res.init(r+rhs.r,c,d);
        for (int i = 0; i < res.c*res.d*res.r; i+=res.c*res.r) {
            for (int j = 0; j < r*c; ++j) {
                res.data[i+j]=data[t1];
                ++t1;
            }
            for (int j = r*c; j < rhs.c*rhs.r+c*r; ++j) {
                res.data[i+j]=rhs.data[t2];
                ++t2;
            }
        }
    }
    if (axis == 1) {
        res.init(r,c+rhs.c,d);
        for (int i = 0; i < res.c*res.d*res.r; i+=res.c) {
            for (int j = 0; j < c; ++j) {
                res.data[i+j]=data[t1];
                ++t1;
            }
            for (int j = c; j < rhs.c+c; ++j) {
                res.data[i+j]=rhs.data[t2];
                ++t2;
            }
        }
    }
    if (axis == 2) {
        res.init(r,c,rhs.d+d);
        for (int i = 0; i < r*c*d ; ++i) {
            res.data[t1]=data[i];
            ++t1;
        }
        for (int i = 0; i < rhs.d*rhs.c*rhs.r; ++i) {
            res.data[t1]=rhs.data[i];
            ++t1;
        }
    }
    return res;
}

Tensor Tensor::convolve(const Tensor &f)const {
  if(!data || !f.data) throw (tensor_not_initialized());
  if(d!=f.d) throw (dimension_mismatch());
  if ((f.r)%2==0 || (f.c)%2==0) throw (filter_odd_dimensions());
  Tensor res(r,c,d);
  int HalfPadRow = ((f.r)-1)/2;
  int HalfPadCol = ((f.c)-1)/2;
  Tensor aux;
  aux = padding(HalfPadRow,HalfPadCol);
  for (int i=HalfPadRow; i<aux.r-HalfPadRow; i++){
    for (int j=HalfPadCol; j<aux.c-HalfPadCol; j++){
      for (int n=0; n<f.d; n++){
        float value = 0;
        for (int l=0; l<f.r; l++){
          for (int m=0; m<f.c; m++){
            value+=aux(i+l-HalfPadRow, j+m-HalfPadCol, n)*f(l, m, n);
          }
        }
        res(i-HalfPadRow, j-HalfPadCol, n) = value;
      }
    }
  }
  return res;
}


int Tensor::rows()const{
    return r;
}

int Tensor::cols()const{
    return c;
}

int Tensor::depth()const{
    return d;
}

float Tensor::getMin(int k)const{
    if (!data) throw (tensor_not_initialized());
    if(k>=d || k<0) throw (index_out_of_bound());
    int sq=r*c;
    float res=data[0];
    for (int i = sq * k; i < sq * (k + 1); ++i) //Parto dall
        if (data[i] < res) res = data[i];
    return res;

}

float Tensor::getMax(int k)const{
    if (!data) throw (tensor_not_initialized());
    if(k>=d || k<0) throw (index_out_of_bound());
    int sq=r*c;
    float res=data[0];
    for (int i = sq * k; i < sq * (k + 1); ++i)
        if (data[i] > res) res = data[i];
    return res;
}

void Tensor::showSize()const{
    cout<<r<<" x "<<c<<" x "<<d<<endl;
}

ostream& operator<<(ostream& stream, const Tensor & obj){
    if(!obj.data) throw (tensor_not_initialized());
    for (int i = 0; i < obj.d*obj.r*obj.c; ++i) {
        if(i%obj.c==0 && i!=0) stream<< endl;
        if(i%(obj.c*obj.r)==0) stream<<endl<<"Depth: "<<i/(obj.r*obj.c)<<endl;
        stream <<" "<< obj.data[i];
    }
    return stream;
}

void Tensor::read_file(string filename) {
    ifstream res(filename);
    if(!res.is_open() || filename.empty()) throw (unable_to_read_file());
    string line;
    getline(res,line);
    r=stoi(line);
    getline(res,line);
    c=stoi(line);
    getline(res,line);
    d=stoi(line);
    delete[] data;
    if(r*c*d==0) data= nullptr;
    else data = new float[r * c * d];
    int i=0;
    while(getline(res,line)) {
        this->data[i] = stof(line);
        ++i;
    }
}

void Tensor::write_file(string filename) {
    ofstream res(filename);
    if(!res.is_open() || filename.empty()) throw (unable_to_read_file());
    res<<r<<"\n"<<c<<"\n"<<d<<"\n";
    for (int i = 0; i < d*c*r; ++i)
        res<<data[i]<<"\n";
    res.close();

}
