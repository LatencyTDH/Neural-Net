3600
====

Projects from AI

This is an implementation of a multilayer feedforward neural net.

Data Sets
------------
###car.data.txt

The Car Evaluation Database decision model evaluates cars according to the following features:

   CAR                      car acceptability
   
   . PRICE                  overall price
   
   . . buying               buying price
   
   . . maint                price of the maintenance
   
   . TECH                   technical characteristics
   
   . . COMFORT              comfort
   
   . . . doors              number of doors
   
   . . . persons            capacity in terms of persons to carry
   
   . . . lug_boot           the size of luggage boot
   
   . . safety               estimated safety of the car
   
and can use the following sets of values:

   CAR              unacc (unacceptable), acc (acceptable), good, v-good (very good)
   
   . PRICE          v-high, high, med, low
   
   . . BUYING       v-high, high, med, low
   
   . . MAINT        v-high, high, med, low
   
   . TECH           poor, satisf, good, v-good
   
   . . COMFORT      bad, acc, good, v-good
   
   . . . DOORS      2, 3, 4, 5-more
   
   . . . PERSONS    2, 4, more
   
   . . . LUG_BOOT   small, med, big
   
   . . SAFETY       low, med, high
   

Input attributes are printed in lowercase. Besides the target concept (CAR), the model includes three intermediate concepts: PRICE, TECH, COMFORT. Every concept in the original model is related to its lower level descendants by a set of examples.

The Car Evaluation Database contains examples with the structural information removed, i.e., directly relates CAR acceptability to the six input attributes: buying, maint, doors, persons, lug_boot, safety. 

Because of known underlying concept structure, this database may be particularly useful for testing constructive induction and structure discovery methods. 

###pendigits.csv

Pen-Based Recognition of Handwritten Digits
Relevant Information:
We create a digit database by collecting 250 samples from 44 writers. The samples written by 30 writers are used for training,cross-validation and writer dependent testing, and the digits written by the other 14 are used for writer independent testing. This database is also available in the UNIPEN format.

**Number of Attributes:**
 16 input+1 class attribute (the locations of a user's pen during the stroke of digits on a sensitive pad)

**For Each Attribute:**
All input attributes are integers in the range 0..100.
The last attribute is the class code 0...9 (the written digit)
