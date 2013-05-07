/*
 * anames.h
 *
 *  Created on: 22-03-2013
 *      Author: Andrzej Biborski
 */
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <vector>
#include <map>

#ifndef ANAMES_H_
#define ANAMES_H_

namespace definitions {
enum Atom {
    vacancy,  H , He , Li , Be , B , C , N , O , F , Ne , Na , Mg , Al , Si , P ,
    S , Cl , Ar , K , Ca , Sc , Ti , V , Cr , Mn , Fe , Co , Ni , Cu , Zn , Ga ,
    Ge , As , Se , Br , Kr , Rb , Sr , Y , Zr , Nb , Mo , Tc , Ru , Rh , Pd , Ag ,
    Cd , In , Sn , Sb , Te , I , Xe , Cs , Ba , La , Ce , Pr , Nd , Pm , Sm , Eu , Gd
    , Tb , Dy , Ho , Er , Tm , Yb , Lu , Hf , Ta , W , Re , Os , Ir , Pt , Au , Hg , Tl ,
    Pb , Bi , Po , At , Rn , Fr , Ra , Ac , Th , Pa , U , Np , Pu , Am , Cm , Bk , Cf , Es ,
    Fm , Md , No , Lr , Rf , Db , Sg , Bh , Hs , Mt , Ds , Rg , Cn , Uut , Fl , Uup , Lv , Uus , Uuo
};

}

namespace utils {

struct AtomMapper {
public:


    AtomMapper(const definitions::Atom * const atoms, int size);
    AtomMapper(const std::vector<definitions::Atom> &atoms);

    definitions::Atom getAtom(int index) const;
    int getIndex(definitions::Atom atom) const;


private:
    int _size;
    std:: vector<definitions::Atom> _atoms;
    std:: map<definitions::Atom,int> _indicies;



    AtomMapper(const AtomMapper&);
    AtomMapper operator=(const AtomMapper&);

    void CreateIndiciesMap();


    void CreateAtomicList(const std::vector<definitions::Atom>& vector) {
        _size = vector.size();
        for(int i = 0; i< _size; ++i)
            _atoms.push_back(vector[i]);
    }


    void CreateAtomicList(const definitions::Atom* const atoms){
        for(int i = 0; i< _size; ++i)
            _atoms.push_back(atoms[i]);
    };


};

}

#endif /* ANAMES_H_ */
