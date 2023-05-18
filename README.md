
____
# UCLA-MSU (CLAMS) Two-Temperature-Model
_____

Code, papers and notes for the UCLA-MSU TTM are stored here.

The TTM model evolves electron and ion temperatures in cylindrical coordinates, consistent with a plasma formed by a laser passing through a dense gas.

We work in SI units everywhere. Some conversions in constants.py

_____________________________
  -physics.py
  	Contains model-independent parameters like Fermi Energy
  	Contains model specfic information as classes of the base parameter class

  -constants.py
  	file containing numerical constants, to be imported with wildcard *
  
  -TTM.py
  	Contains two-temperature model solver and experiment classes

