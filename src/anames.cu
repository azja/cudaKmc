#include "../headers/anames.h"



namespace utils {

void AtomMapper::CreateIndiciesMap() {
	for(int i = 0; i < _size;++i){
		_indicies.insert(std::pair<definitions::Atom,int>(_atoms[i],i));
	}
}


AtomMapper::AtomMapper(const std::vector<definitions::Atom>& atoms) {
	CreateAtomicList(atoms);
	CreateIndiciesMap();
}


AtomMapper::AtomMapper(const definitions::Atom* const atoms, int size):_size(size) {
	CreateAtomicList(atoms);
	CreateIndiciesMap();
}
inline definitions::Atom AtomMapper::getAtom(int index) const {
	 return _atoms[index];
}


inline int AtomMapper::getIndex(definitions::Atom atom) const {
   return _indicies.at(atom);
}

}
