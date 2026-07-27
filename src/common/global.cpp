#include    "../../include/common/environment.hpp"

namespace isdf{

std::ofstream  isdfOFS;
std::ofstream  psiOFS;
void ErrorHandling( const char * msg ){
  isdfOFS << std::endl << "ERROR!" << std::endl 
    << msg << std::endl << std::endl;
  throw std::runtime_error( msg );
}

} // namespace isdf
