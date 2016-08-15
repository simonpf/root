// @(#)root/tmva $Id$
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodDNN                                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      A neural network implementation                                           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer      <peter.speckmayer@gmx.ch> - CERN, Switzerland       *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//
// neural network implementation
//_______________________________________________________________________

#include "TString.h"
#include "TTree.h"
#include "TFile.h"
#include "TFormula.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/MethodDNN.h"
#include "TMVA/Timer.h"
#include "TMVA/Types.h"
#include "TMVA/Tools.h"
#include "TMVA/Config.h"
#include "TMVA/Ranking.h"

#include "TMVA/DNN/Net.h"
#include "TMVA/DNN/Architectures/Reference.h"

#include "TMVA/NeuralNet.h"
#include "TMVA/Monitoring.h"

#include <algorithm>
#include <iostream>

REGISTER_METHOD(DNN)

ClassImp(TMVA::MethodDNN)

using TMVA::DNN::EActivationFunction;
using TMVA::DNN::ELossFunction;
using TMVA::DNN::EInitialization;
using TMVA::DNN::EOutputFunction;


   namespace TMVA
   {
      namespace DNN
      {
         template <typename Container, typename T>
         void gaussDistribution (Container& container, T mean, T sigma)
         {
            for (auto it = begin (container), itEnd = end (container); it != itEnd; ++it)
               {
                  (*it) = DNN::gaussDouble (mean, sigma);
               }
         }
      };
   };






//______________________________________________________________________________
TMVA::MethodDNN::MethodDNN( const TString& jobName,
                            const TString& methodTitle,
                            DataSetInfo& theData,
                            const TString& theOption )
   : MethodBase( jobName, Types::kDNN, methodTitle, theData, theOption)
   , fResume (false)
{
   // standard constructor
}

//______________________________________________________________________________
TMVA::MethodDNN::MethodDNN( DataSetInfo& theData,
                            const TString& theWeightFile)
   : MethodBase( Types::kDNN, theData, theWeightFile)
   , fResume (false)
{
   // constructor from a weight file
}

//______________________________________________________________________________
TMVA::MethodDNN::~MethodDNN()
{
   // destructor
   // nothing to be done
}

//_______________________________________________________________________
Bool_t TMVA::MethodDNN::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/ )
{
   // MLP can handle classification with 2 classes and regression with one regression-target
   if (type == Types::kClassification && numberClasses == 2 ) return kTRUE;
   if (type == Types::kMulticlass ) return kTRUE;
   if (type == Types::kRegression ) return kTRUE;

   return kFALSE;
}

//______________________________________________________________________________
void TMVA::MethodDNN::Init()
{
   // default initializations
}

//_______________________________________________________________________
void TMVA::MethodDNN::DeclareOptions()
{
   // define the options (their key words) that can be set in the option string
   // know options:
   // TrainingMethod  <string>     Training method
   //    available values are:         BP   Back-Propagation <default>
   //                                  GA   Genetic Algorithm (takes a LONG time)
   //
   // LearningRate    <float>      DNN learning rate parameter
   // DecayRate       <float>      Decay rate for learning parameter
   // TestRate        <int>        Test for overtraining performed at each #th epochs
   //
   // BPMode          <string>     Back-propagation learning mode
   //    available values are:         sequential <default>
   //                                  batch
   //
   // BatchSize       <int>        Batch size: number of events/batch, only set if in Batch Mode,
   //                                          -1 for BatchSize=number_of_events

   // DeclareOptionRef(fTrainMethodS="SD", "TrainingMethod",
   //                  "Train with back propagation steepest descend");
   // AddPreDefVal(TString("SD"));

   //   DeclareOptionRef(fLayoutString="TANH|(N+30)*2,TANH|(N+30),LINEAR",    "Layout",    "neural network layout");
   // DeclareOptionRef(fLayoutString="RELU|(N+20)*2,RELU|(N+10)*2,LINEAR",    "Layout",    "neural network layout");
   DeclareOptionRef(fLayoutString="SOFTSIGN|(N+100)*2,LINEAR",    "Layout",    "neural network layout");


   DeclareOptionRef(fErrorStrategy="CROSSENTROPY",    "ErrorStrategy",    "error strategy (regression: sum of squares; classification: crossentropy; multiclass: crossentropy/mutual exclusive cross entropy");
   AddPreDefVal(TString("CROSSENTROPY"));
   AddPreDefVal(TString("SUMOFSQUARES"));
   AddPreDefVal(TString("MUTUALEXCLUSIVE"));
   AddPreDefVal(TString("CHECKGRADIENTS"));


   DeclareOptionRef(fWeightInitializationStrategyString="XAVIER",    "WeightInitialization",    "Weight initialization strategy");
   AddPreDefVal(TString("XAVIER"));
   AddPreDefVal(TString("XAVIERUNIFORM"));
   AddPreDefVal(TString("LAYERSIZE"));

   DeclareOptionRef(fGPUString="True", "GPU", "Use GPU for training.");

   DeclareOptionRef(fTrainingStrategy="LearningRate=1e-1,Momentum=0.3,Repetitions=3,ConvergenceSteps=50,BatchSize=30,TestRepetitions=7,WeightDecay=0.0,Renormalize=L2,DropConfig=0.0,DropRepetitions=5|LearningRate=1e-4,Momentum=0.3,Repetitions=3,ConvergenceSteps=50,BatchSize=20,TestRepetitions=7,WeightDecay=0.001,Renormalize=L2,DropConfig=0.0+0.5+0.5,DropRepetitions=5,Multithreading=True",    "TrainingStrategy",    "defines the training strategies");

   DeclareOptionRef(fSumOfSigWeights_test=1000.0,    "SignalWeightsSum",    "Sum of weights of signal; Is used to compute the significance on the fly");
   DeclareOptionRef(fSumOfBkgWeights_test=1000.0,    "BackgroundWeightsSum",    "Sum of weights of background; Is used to compute the significance on the fly");
}


std::vector<std::pair<int,TMVA::DNN::EnumFunction>> TMVA::MethodDNN::ParseLayoutString(TString layerSpec)
{
   // parse layout specification string and return a vector, each entry
   // containing the number of neurons to go in each successive layer
   std::vector<std::pair<int,TMVA::DNN::EnumFunction>> layout;
   const TString delim_Layer (",");
   const TString delim_Sub ("|");

   const size_t inputSize = GetNvar ();

   TObjArray* layerStrings = layerSpec.Tokenize (delim_Layer);
   TIter nextLayer (layerStrings);
   TObjString* layerString = (TObjString*)nextLayer ();
   for (; layerString != NULL; layerString = (TObjString*)nextLayer ())
      {
         int numNodes = 0;
         TMVA::DNN::EnumFunction eActivationFunction = DNN::EnumFunction::TANH;

         TObjArray* subStrings = layerString->GetString ().Tokenize (delim_Sub);
         TIter nextToken (subStrings);
         TObjString* token = (TObjString*)nextToken ();
         int idxToken = 0;
         for (; token != NULL; token = (TObjString*)nextToken ())
            {
               switch (idxToken)
                  {
                  case 0:
                     {
                        TString strActFnc (token->GetString ());
                        if (strActFnc == "RELU")
                           eActivationFunction = DNN::EnumFunction::RELU;
                        else if (strActFnc == "TANH")
                           eActivationFunction = DNN::EnumFunction::TANH;
                        else if (strActFnc == "SYMMRELU")
                           eActivationFunction = DNN::EnumFunction::SYMMRELU;
                        else if (strActFnc == "SOFTSIGN")
                           eActivationFunction = DNN::EnumFunction::SOFTSIGN;
                        else if (strActFnc == "SIGMOID")
                           eActivationFunction = DNN::EnumFunction::SIGMOID;
                        else if (strActFnc == "LINEAR")
                           eActivationFunction = DNN::EnumFunction::LINEAR;
                        else if (strActFnc == "GAUSS")
                           eActivationFunction = DNN::EnumFunction::GAUSS;
                     }
                     break;
                  case 1: // number of nodes
                     {
                        TString strNumNodes (token->GetString ());
                        TString strN ("x");
                        strNumNodes.ReplaceAll ("N", strN);
                        strNumNodes.ReplaceAll ("n", strN);
                        TFormula fml ("tmp",strNumNodes);
                        numNodes = fml.Eval (inputSize);
                     }
                     break;
                  }
               ++idxToken;
            }
         layout.push_back (std::make_pair (numNodes,eActivationFunction));
      }
   return layout;
}



// parse key value pairs in blocks -> return vector of blocks with map of key value pairs
std::vector<std::map<TString,TString>> TMVA::MethodDNN::ParseKeyValueString(TString parseString, TString blockDelim, TString tokenDelim)
{
   std::vector<std::map<TString,TString>> blockKeyValues;
   const TString keyValueDelim ("=");

   //    const size_t inputSize = GetNvar ();

   TObjArray* blockStrings = parseString.Tokenize (blockDelim);
   TIter nextBlock (blockStrings);
   TObjString* blockString = (TObjString*)nextBlock ();
   for (; blockString != NULL; blockString = (TObjString*)nextBlock ())
      {
         blockKeyValues.push_back (std::map<TString,TString> ()); // new block
         std::map<TString,TString>& currentBlock = blockKeyValues.back ();

         TObjArray* subStrings = blockString->GetString ().Tokenize (tokenDelim);
         TIter nextToken (subStrings);
         TObjString* token = (TObjString*)nextToken ();
       
         for (; token != NULL; token = (TObjString*)nextToken ())
            {
               TString strKeyValue (token->GetString ());
               int delimPos = strKeyValue.First (keyValueDelim.Data ());
               if (delimPos <= 0)
                  continue;

               TString strKey = TString (strKeyValue (0, delimPos));
               strKey.ToUpper ();
               TString strValue = TString (strKeyValue (delimPos+1, strKeyValue.Length ()));

               strKey.Strip (TString::kBoth, ' ');
               strValue.Strip (TString::kBoth, ' ');

               currentBlock.insert (std::make_pair (strKey, strValue));
            }
      }
   return blockKeyValues;
}


TString fetchValue (const std::map<TString, TString>& keyValueMap, TString _key)
{
   TString key (_key);
   key.ToUpper ();
   std::map<TString, TString>::const_iterator it = keyValueMap.find (key);
   if (it == keyValueMap.end ())
      return TString ("");
   return it->second;
}

template <typename T>
T fetchValue (const std::map<TString,TString>& keyValueMap, TString key, T defaultValue);

template <>
int fetchValue (const std::map<TString,TString>& keyValueMap, TString key, int defaultValue)
{
   TString value (fetchValue (keyValueMap, key));
   if (value == "")
      return defaultValue;
   return value.Atoi ();
}

template <>
double fetchValue (const std::map<TString,TString>& keyValueMap, TString key, double defaultValue)
{
   TString value (fetchValue (keyValueMap, key));
   if (value == "")
      return defaultValue;
   return value.Atof ();
}

template <>
TString fetchValue (const std::map<TString,TString>& keyValueMap, TString key, TString defaultValue)
{
   TString value (fetchValue (keyValueMap, key));
   if (value == "")
      return defaultValue;
   return value;
}

template <>
bool fetchValue (const std::map<TString,TString>& keyValueMap, TString key, bool defaultValue)
{
   TString value (fetchValue (keyValueMap, key));
   if (value == "")
      return defaultValue;
   value.ToUpper ();
   if (value == "TRUE" ||
       value == "T" ||
       value == "1")
      return true;
   return false;
}

template <>
std::vector<double> fetchValue (const std::map<TString,TString>& keyValueMap, TString key, std::vector<double> defaultValue)
{
   TString parseString (fetchValue (keyValueMap, key));
   if (parseString == "")
      return defaultValue;
   parseString.ToUpper ();
   std::vector<double> values;

   const TString tokenDelim ("+");
   TObjArray* tokenStrings = parseString.Tokenize (tokenDelim);
   TIter nextToken (tokenStrings);
   TObjString* tokenString = (TObjString*)nextToken ();
   for (; tokenString != NULL; tokenString = (TObjString*)nextToken ())
      {
         std::stringstream sstr;
         double currentValue;
         sstr << tokenString->GetString ().Data ();
         sstr >> currentValue;
         values.push_back (currentValue);
      }
   return values;
}



//_______________________________________________________________________
void TMVA::MethodDNN::ProcessOptions()
{
   // process user options
   //   MethodBase::ProcessOptions();

   if (fErrorStrategy == "CHECKGRADIENTS") 
      return checkGradients ();


   
   if (IgnoreEventsWithNegWeightsInTraining()) {
      Log() << kINFO 
            << "Will ignore negative events in training!"
            << Endl;
   }

   fLayout = TMVA::MethodDNN::ParseLayoutString (fLayoutString);

   //                                                                                         block-delimiter  token-delimiter
   std::vector<std::map<TString,TString>> strategyKeyValues = ParseKeyValueString (fTrainingStrategy, TString ("|"), TString (","));


   if (fWeightInitializationStrategyString == "XAVIER")
      fWeightInitializationStrategy = TMVA::DNN::WeightInitializationStrategy::XAVIER;
   if (fWeightInitializationStrategyString == "XAVIERUNIFORM")
      fWeightInitializationStrategy = TMVA::DNN::WeightInitializationStrategy::XAVIERUNIFORM;
   else if (fWeightInitializationStrategyString == "LAYERSIZE")
      fWeightInitializationStrategy = TMVA::DNN::WeightInitializationStrategy::LAYERSIZE;
   else if (fWeightInitializationStrategyString == "TEST")
      fWeightInitializationStrategy = TMVA::DNN::WeightInitializationStrategy::TEST;
   else
      fWeightInitializationStrategy = TMVA::DNN::WeightInitializationStrategy::XAVIER;

   fGPUString.ToUpper ();
   if (fGPUString.BeginsWith ("T"))
       fGPU = true;
   else
       fGPU = false;

   // create settings
   if (fAnalysisType == Types::kClassification)
      {
         if (fErrorStrategy == "SUMOFSQUARES") fModeErrorFunction = TMVA::DNN::ModeErrorFunction::SUMOFSQUARES;
         if (fErrorStrategy == "CROSSENTROPY") fModeErrorFunction = TMVA::DNN::ModeErrorFunction::CROSSENTROPY;
         if (fErrorStrategy == "MUTUALEXCLUSIVE") fModeErrorFunction = TMVA::DNN::ModeErrorFunction::CROSSENTROPY_MUTUALEXCLUSIVE;
      }
   else if (fAnalysisType == Types::kMulticlass)
      {
         if (fErrorStrategy == "SUMOFSQUARES") fModeErrorFunction = TMVA::DNN::ModeErrorFunction::SUMOFSQUARES;
         if (fErrorStrategy == "CROSSENTROPY") fModeErrorFunction = TMVA::DNN::ModeErrorFunction::CROSSENTROPY;
         if (fErrorStrategy == "MUTUALEXCLUSIVE") fModeErrorFunction = TMVA::DNN::ModeErrorFunction::CROSSENTROPY_MUTUALEXCLUSIVE;
      }
   else if (fAnalysisType == Types::kRegression)
      {
         if (fErrorStrategy != "SUMOFSQUARES")
            {
               Log () << kWARNING 
                      << "For regression only SUMOFSQUARES is a valid neural net error function."
                      << "Setting error function to SUMOFSQUARES now."
                      << Endl;
            }
         fModeErrorFunction = TMVA::DNN::ModeErrorFunction::SUMOFSQUARES;
      }
   
   for (auto& block : strategyKeyValues)
      {
         size_t convergenceSteps = fetchValue (block, "ConvergenceSteps", 100);
         int batchSize = fetchValue (block, "BatchSize", 30);
         int testRepetitions = fetchValue (block, "TestRepetitions", 7);
         double factorWeightDecay = fetchValue (block, "WeightDecay", 0.0);
         TString regularization = fetchValue (block, "Regularization", TString ("NONE"));
         double learningRate = fetchValue (block, "LearningRate", 1e-5);
         double momentum = fetchValue (block, "Momentum", 0.3);
         int repetitions = fetchValue (block, "Repetitions", 3);
         TString strMultithreading = fetchValue (block, "Multithreading", TString ("True"));
         std::vector<double> dropConfig;
         dropConfig = fetchValue (block, "DropConfig", dropConfig);
         int dropRepetitions = fetchValue (block, "DropRepetitions", 3);

         TMVA::DNN::EnumRegularization eRegularization = TMVA::DNN::EnumRegularization::NONE;
         if (regularization == "L1")
            eRegularization = TMVA::DNN::EnumRegularization::L1;
         else if (regularization == "L2")
            eRegularization = TMVA::DNN::EnumRegularization::L2;
         else if (regularization == "L1MAX")
            eRegularization = TMVA::DNN::EnumRegularization::L1MAX;


         strMultithreading.ToUpper ();
         bool multithreading = true;
         if (strMultithreading.BeginsWith ("T"))
            multithreading = true;
         else
            multithreading = false;
           

         if (fAnalysisType == Types::kClassification)
            {
               std::shared_ptr<TMVA::DNN::ClassificationSettings> ptrSettings = make_shared <TMVA::DNN::ClassificationSettings> (
                                                                                                                                 GetName  (),
                                                                                                                                 convergenceSteps, batchSize, 
                                                                                                                                 testRepetitions, factorWeightDecay,
                                                                                                                                 eRegularization, fScaleToNumEvents, TMVA::DNN::MinimizerType::fSteepest,
                                                                                                                                 learningRate, 
                                                                                                                                 momentum, repetitions, multithreading);
               ptrSettings->setWeightSums (fSumOfSigWeights_test, fSumOfBkgWeights_test);
               fSettings.push_back (ptrSettings);
            }
         else if (fAnalysisType == Types::kMulticlass)
            {
               std::shared_ptr<TMVA::DNN::Settings> ptrSettings = make_shared <TMVA::DNN::Settings> (
                                                                                                     GetName  (),
                                                                                                     convergenceSteps, batchSize, 
                                                                                                     testRepetitions, factorWeightDecay,
                                                                                                     eRegularization, TMVA::DNN::MinimizerType::fSteepest,
                                                                                                     learningRate, 
                                                                                                     momentum, repetitions, multithreading);
               fSettings.push_back (ptrSettings);
            }
         else if (fAnalysisType == Types::kRegression)
            {
               std::shared_ptr<TMVA::DNN::Settings> ptrSettings = make_shared <TMVA::DNN::Settings> (
                                                                                                     GetName  (),
                                                                                                     convergenceSteps, batchSize, 
                                                                                                     testRepetitions, factorWeightDecay,
                                                                                                     eRegularization, TMVA::DNN::MinimizerType::fSteepest,
                                                                                                     learningRate, 
                                                                                                     momentum, repetitions, multithreading);
               fSettings.push_back (ptrSettings);
            }

           
         if (dropRepetitions > 0 && !dropConfig.empty ())
            {
               fSettings.back ()->setDropOut (std::begin (dropConfig), std::end (dropConfig), dropRepetitions);
            }
           
      }
}

//______________________________________________________________________________
void TMVA::MethodDNN::Train()
{
    if (fGPU) {
        TrainGPU();
    } else {

       fMonitoring = NULL;
       // if (!fMonitoring)
       // {
       //     fMonitoring = make_shared<Monitoring>();
       //     fMonitoring->Start ();
       // }

       // INITIALIZATION
       // create pattern
       std::vector<Pattern> trainPattern;
       std::vector<Pattern> testPattern;

       const std::vector<TMVA::Event*>& eventCollectionTraining = GetEventCollection (Types::kTraining);
       const std::vector<TMVA::Event*>& eventCollectionTesting  = GetEventCollection (Types::kTesting);

       for (size_t iEvt = 0, iEvtEnd = eventCollectionTraining.size (); iEvt < iEvtEnd; ++iEvt)
       {
          const TMVA::Event* event = eventCollectionTraining.at (iEvt);
          const std::vector<Float_t>& values  = event->GetValues  ();
          if (fAnalysisType == Types::kClassification)
          {
             double outputValue = event->GetClass () == 0 ? 0.9 : 0.1;
             trainPattern.push_back (Pattern (values.begin  (), values.end (), outputValue, event->GetWeight ()));
             trainPattern.back ().addInput (1.0); // bias node
          }
          else
          {
             const std::vector<Float_t>& targets = event->GetTargets ();
             trainPattern.push_back (Pattern (values.begin  (), values.end (), targets.begin (), targets.end (), event->GetWeight ()));
             trainPattern.back ().addInput (1.0); // bias node
          }
       }

       for (size_t iEvt = 0, iEvtEnd = eventCollectionTesting.size (); iEvt < iEvtEnd; ++iEvt)
       {
          const TMVA::Event* event = eventCollectionTesting.at (iEvt);
          const std::vector<Float_t>& values  = event->GetValues  ();
          if (fAnalysisType == Types::kClassification)
          {
             double outputValue = event->GetClass () == 0 ? 0.9 : 0.1;
             testPattern.push_back (Pattern (values.begin  (), values.end (), outputValue, event->GetWeight ()));
             testPattern.back ().addInput (1.0); // bias node
          }
          else
          {
             const std::vector<Float_t>& targets = event->GetTargets ();
             testPattern.push_back (Pattern (values.begin  (), values.end (), targets.begin (), targets.end (), event->GetWeight ()));
             testPattern.back ().addInput (1.0); // bias node
          }
       }

       if (trainPattern.empty () || testPattern.empty ())
           return;

       // create net and weights
       fNet.clear ();
       fWeights.clear ();

       // if "resume" from saved weights
       if (fResume)
       {
          std::cout << ".. resume" << std::endl;
          //        std::tie (fNet, fWeights) = ReadWeights (fFileName);
       }
       else // initialize weights and net
       {
          size_t inputSize = GetNVariables (); //trainPattern.front ().input ().size ();
          size_t outputSize = fAnalysisType == Types::kClassification ? 1 : GetNTargets (); //trainPattern.front ().output ().size ();
          fNet.setInputSize (inputSize + 1); // num vars + bias node
          fNet.setOutputSize (outputSize); // num vars + bias node
        
          // configure neural net
          auto itLayout = std::begin (fLayout), itLayoutEnd = std::end (fLayout)-1; // all layers except the last one
          for ( ; itLayout != itLayoutEnd; ++itLayout)
          {
             fNet.addLayer (DNN::Layer ((*itLayout).first, (*itLayout).second)); 
             Log() << kINFO 
                   << "Add Layer with " << (*itLayout).first << " nodes." 
                   << Endl;
          }

          DNN::ModeOutputValues eModeOutputValues = DNN::ModeOutputValues::SIGMOID;
          if (fAnalysisType == Types::kRegression)
          {
             eModeOutputValues = DNN::ModeOutputValues::DIRECT;
          }
          else if ((fAnalysisType == Types::kClassification ||
                    fAnalysisType == Types::kMulticlass) &&
                   fModeErrorFunction == TMVA::DNN::ModeErrorFunction::SUMOFSQUARES)
          {
             eModeOutputValues = DNN::ModeOutputValues::DIRECT;
          }
          fNet.addLayer (DNN::Layer (outputSize, (*itLayout).second, eModeOutputValues)); 
          Log() << kINFO 
                << "Add Layer with " << outputSize << " nodes." 
                << Endl << Endl;
          fNet.setErrorFunction (fModeErrorFunction); 

          size_t numWeights = fNet.numWeights ();
          Log() << kINFO 
                << "Total number of Synapses = " 
                << numWeights
                << Endl;

          // initialize weights
          fNet.initializeWeights (fWeightInitializationStrategy, 
                                  std::back_inserter (fWeights));
       }


       // loop through settings 
       // and create "settings" and minimizer 
       int idxSetting = 0;
       for (auto itSettings = std::begin (fSettings), itSettingsEnd = std::end (fSettings); itSettings != itSettingsEnd; ++itSettings, ++idxSetting)
       {
          std::shared_ptr<TMVA::DNN::Settings> ptrSettings = *itSettings;
          ptrSettings->setMonitoring (fMonitoring);
          Log() << kINFO
                << "Training with learning rate = " << ptrSettings->learningRate ()
                << ", momentum = " << ptrSettings->momentum ()
                << ", repetitions = " << ptrSettings->repetitions ()
                << Endl;

          ptrSettings->setProgressLimits ((idxSetting)*100.0/(fSettings.size ()), (idxSetting+1)*100.0/(fSettings.size ()));

          const std::vector<double>& dropConfig = ptrSettings->dropFractions ();
          if (!dropConfig.empty ())
          {
             Log () << kINFO << "Drop configuration" << Endl
                    << "    drop repetitions = " << ptrSettings->dropRepetitions () << Endl;
          }
          int idx = 0;
          for (auto f : dropConfig)
          {
             Log () << kINFO << "    Layer " << idx << " = " << f << Endl;
             ++idx;
          }
          Log () << kINFO << Endl;
        
          if (ptrSettings->minimizerType () == TMVA::DNN::MinimizerType::fSteepest)
          {
             DNN::Steepest minimizer (ptrSettings->learningRate (), ptrSettings->momentum (), ptrSettings->repetitions ());
             /*E =*/fNet.train (fWeights, trainPattern, testPattern, minimizer, *ptrSettings.get ());
          }
          ptrSettings.reset ();
          Log () << kINFO << Endl;
       }
       fMonitoring = 0;
    }
}

void TMVA::MethodDNN::TrainGPU()
{

#ifdef DNNCUDA // Included only if DNNCUDA flag is set.

   TMVA::DNN::TNet<TMVA::DNN::TCuda> GPUNet{};

   size_t inputSize = GetNVariables ();
   size_t outputSize = (GetNTargets() == 0) ? 1 : GetNTargets();

   GPUNet.SetInputWidth(inputSize);

   // Also need to set standard net structure.
   fNet.setInputSize (inputSize + 1);
   fNet.setOutputSize (outputSize);

   // configure neural net
   auto itLayout = std::begin (fLayout), itLayoutEnd = std::end (fLayout)-1; // all layers except the last one
   for ( ; itLayout != itLayoutEnd; ++itLayout)
   {
      fNet.addLayer (DNN::Layer ((*itLayout).first, (*itLayout).second)); 
      TMVA::DNN::EnumFunction f = (*itLayout).second;
      switch(f)
      {
      case DNN::EnumFunction::RELU :
          GPUNet.AddLayer((*itLayout).first, EActivationFunction::RELU);
          break;
      case DNN::EnumFunction::TANH :
          GPUNet.AddLayer((*itLayout).first, EActivationFunction::TANH);
          break;
      case DNN::EnumFunction::SYMMRELU :
          GPUNet.AddLayer((*itLayout).first, EActivationFunction::SYMMRELU);
          break;
      case DNN::EnumFunction::SOFTSIGN :
          GPUNet.AddLayer((*itLayout).first, EActivationFunction::SOFTSIGN);
          break;
      case DNN::EnumFunction::SIGMOID :
          GPUNet.AddLayer((*itLayout).first, EActivationFunction::SIGMOID);
          break;
      case DNN::EnumFunction::LINEAR :
          GPUNet.AddLayer((*itLayout).first, EActivationFunction::IDENTITY);
          break;
      case DNN::EnumFunction::GAUSS :
          GPUNet.AddLayer((*itLayout).first, EActivationFunction::GAUSS);
          break;
      default :
          GPUNet.AddLayer((*itLayout).first, EActivationFunction::IDENTITY);
          break;
      }
   }

   DNN::ModeOutputValues eModeOutputValues = DNN::ModeOutputValues::SIGMOID;
   if (fAnalysisType == Types::kRegression)
   {
      eModeOutputValues = DNN::ModeOutputValues::DIRECT;
      GPUNet.AddLayer(outputSize, EActivationFunction::IDENTITY);
      GPUNet.SetLossFunction(ELossFunction::MEANSQUAREDERROR);
   } else if ((fAnalysisType == Types::kClassification ||
               fAnalysisType == Types::kMulticlass) &&
              fModeErrorFunction == TMVA::DNN::ModeErrorFunction::SUMOFSQUARES) {
      GPUNet.AddLayer(outputSize, EActivationFunction::IDENTITY);
      GPUNet.SetLossFunction(ELossFunction::MEANSQUAREDERROR);
   } else {
      eModeOutputValues = DNN::ModeOutputValues::DIRECT;
      GPUNet.AddLayer(outputSize, EActivationFunction::IDENTITY);
      GPUNet.SetLossFunction(ELossFunction::CROSSENTROPY);
   }

   fNet.addLayer (DNN::Layer (outputSize, (*itLayout).second, eModeOutputValues));
   fNet.setErrorFunction (fModeErrorFunction);

   switch(fWeightInitializationStrategy)
   {
   case DNN::WeightInitializationStrategy::XAVIER :
       GPUNet.Initialize(EInitialization::GAUSS);
       break;
   case DNN::WeightInitializationStrategy::XAVIERUNIFORM:
       GPUNet.Initialize(EInitialization::UNIFORM);
       break;
   default :
       GPUNet.Initialize(EInitialization::GAUSS);
       break;
   }

   size_t nTrainingSamples = GetEventCollection(Types::kTraining).size();
   size_t nTestSamples     = GetEventCollection(Types::kTesting).size();

   int idxSetting = 0;
   for (auto itSettings = std::begin (fSettings), itSettingsEnd = std::end (fSettings);
        itSettings != itSettingsEnd; ++itSettings, ++idxSetting)
      {

         TMVA::DNN::Settings settings = **itSettings;
         settings.setMonitoring (fMonitoring);

         Log() << kINFO
               << "Training on GPU with learning rate = "
               << settings.learningRate ()
               << ", momentum = " << settings.momentum ()
               << ", repetitions = " << settings.repetitions ()
               << Endl;

         settings.setProgressLimits ((idxSetting)*100.0/(fSettings.size ()),
                                         (idxSetting+1)*100.0/(fSettings.size ()));

         const std::vector<double>& dropConfig = settings.dropFractions ();
         if (!dropConfig.empty ())
         {
            Log () << kINFO << "Drop configuration" << Endl
                   << "    drop repetitions = "
                   << settings.dropRepetitions () << Endl;
         }

         auto trainNet = GPUNet.CreateClone(settings.batchSize());
         int idx = 0;
         for (auto f : dropConfig)
         {
            Log () << kINFO << "    Layer " << idx << " = " << f << Endl;
            trainNet.GetLayer(idx).SetDropoutProbability(f);
            ++idx;
         }
         Log () << kINFO << Endl;

         using DataLoader_t = typename DNN::TCuda::DataLoader_t<DNN::TMVAInput_t>;
         DataLoader_t trainingData(GetEventCollection(Types::kTraining),
                                   nTrainingSamples,
                                   trainNet.GetBatchSize(),
                                   trainNet.GetInputWidth(),
                                   trainNet.GetOutputWidth());

         DataLoader_t testData(GetEventCollection(Types::kTesting),
                               nTestSamples,
                               nTestSamples,
                               trainNet.GetInputWidth(),
                               trainNet.GetOutputWidth());
         auto testNet   = GPUNet.CreateClone(testData.GetBatchSize());
         DNN::TGradientDescent<DNN::TCuda> minimizer{};

         minimizer.Reset();
         minimizer.SetLearningRate(settings.learningRate());
         minimizer.SetTestInterval(settings.testRepetitions());
         minimizer.SetConvergenceSteps(settings.convergenceSteps());

         bool converged = false;
         size_t stepCount = 0;

         while (!converged)
         {
            // Perform minimization steps for a full epoch.
            if ((stepCount % minimizer.GetTestInterval()) != 0) {
               for (auto batch : trainingData) {
                  auto inputMatrix  = batch.GetInput();
                  auto outputMatrix = batch.GetOutput();
                  minimizer.StepReducedWeights(trainNet, inputMatrix, outputMatrix);
               }
            } else {
               Double_t trainingError = 0.0;
               for (auto batch : trainingData) {
                  auto inputMatrix  = batch.GetInput();
                  auto outputMatrix = batch.GetOutput();
                  trainingError += minimizer.StepReducedWeightsLoss(
                      trainNet,
                      inputMatrix,
                      outputMatrix);
               }
               trainingError /= (Double_t) trainingData.GetNBatchesInEpoch();

               auto testBatch  = *testData.begin();
               auto testInput  = testBatch.GetInput();
               auto testOutput = testBatch.GetOutput();
               minimizer.TestError(testNet, testInput, testOutput);

               TString convText = Form("(train/test/epo/conv/maxco): %.3g/%.3g/%d/%d",
                                       trainingError,
                                       minimizer.GetTestError(),
                                       (int) stepCount,
                                       (int) minimizer.GetConvergenceCount ());
               Double_t progress = minimizer.GetConvergenceCount()
                                   / settings.convergenceSteps();
               settings.cycle(progress, convText);
               converged = minimizer.HasConverged();
            }
            stepCount++;
         }
         fMonitoring = 0;
      }
   fWeights.clear();

   size_t weightIndex = 0;
   size_t prevLayerWidth = GPUNet.GetInputWidth();

   for (size_t l = 0; l < GPUNet.GetDepth(); l++) {
      auto &layer = GPUNet.GetLayer(l);
      size_t layerWidth = layer.GetWidth();
      size_t layerSize = prevLayerWidth * layerWidth;
      fWeights.reserve(fWeights.size() + layerSize);
      TMatrixT<Double_t> Weights(layer.GetWeights());

      for (size_t j = 0; j < (size_t) Weights.GetNcols(); j++) {
         for (size_t i = 0; i < (size_t) Weights.GetNrows(); i++) {
            fWeights.push_back(Weights(i,j));
         }
      }

      if (l == 0) {
         fWeights.reserve(fWeights.size() + layerWidth);
         TMatrixT<Double_t> theta(layer.GetBiases());
         for (size_t i = 0; i < layerWidth; i++) {
             fWeights.push_back(theta(i,0));
         }
      }
      prevLayerWidth = layerWidth;
   }

#else // DNNCUDA flag not set.

   Log() << kFATAL << "CUDA backend not enabled. Please make sure "
                      "you have CUDA installed and it was successfully "
                      "detected by CMAKE." << Endl;
#endif // DNNCUDA
}

//_______________________________________________________________________
Double_t TMVA::MethodDNN::GetMvaValue( Double_t* /*errLower*/, Double_t* /*errUpper*/ )
{
   if (fWeights.empty ())
      return 0.0;

   const std::vector<Float_t>& inputValues = GetEvent ()->GetValues ();
   std::vector<double> input (inputValues.begin (), inputValues.end ());
   input.push_back (1.0); // bias node
   std::vector<double> output = fNet.compute (input, fWeights);
   if (output.empty ())
      return 0.0;

   return output.at (0);
}

////////////////////////////////////////////////////////////////////////////////
/// get the regression value generated by the DNN

const std::vector<Float_t> &TMVA::MethodDNN::GetRegressionValues() 
{
   assert (!fWeights.empty ());
   if (fWeights.empty ())
      return *fRegressionReturnVal;

   const Event * ev = GetEvent();
    
   const std::vector<Float_t>& inputValues = ev->GetValues ();
   std::vector<double> input (inputValues.begin (), inputValues.end ());
   input.push_back (1.0); // bias node
   std::vector<double> output = fNet.compute (input, fWeights);

   if (fRegressionReturnVal == NULL) fRegressionReturnVal = new std::vector<Float_t>();
   fRegressionReturnVal->clear();

   assert (!output.empty ());
   if (output.empty ())
      return *fRegressionReturnVal;

   Event * evT = new Event(*ev);
   UInt_t ntgts = fNet.outputSize ();
   for (UInt_t itgt = 0; itgt < ntgts; ++itgt) {
      evT->SetTarget(itgt,output.at (itgt));
   }

   const Event* evT2 = GetTransformationHandler().InverseTransform( evT );
   for (UInt_t itgt = 0; itgt < ntgts; ++itgt) {
      fRegressionReturnVal->push_back( evT2->GetTarget(itgt) );
   }

   delete evT;

   return *fRegressionReturnVal;
}









////////////////////////////////////////////////////////////////////////////////
/// get the multiclass classification values generated by the DNN

const std::vector<Float_t> &TMVA::MethodDNN::GetMulticlassValues()
{
   if (fWeights.empty ())
      return *fRegressionReturnVal;

   const std::vector<Float_t>& inputValues = GetEvent ()->GetValues ();
   std::vector<double> input (inputValues.begin (), inputValues.end ());
   input.push_back (1.0); // bias node
   std::vector<double> output = fNet.compute (input, fWeights);

   // check the output of the network
 
   if (fMulticlassReturnVal == NULL) fMulticlassReturnVal = new std::vector<Float_t>();
   fMulticlassReturnVal->clear();
   std::vector<Float_t> temp;

   UInt_t nClasses = DataInfo().GetNClasses();
   assert (nClasses == output.size());
   for (UInt_t icls = 0; icls < nClasses; icls++) {
      temp.push_back (output.at (icls));
   }
   
   for(UInt_t iClass=0; iClass<nClasses; iClass++){
      Double_t norm = 0.0;
      for(UInt_t j=0;j<nClasses;j++){
         if(iClass!=j)
            norm+=exp(temp[j]-temp[iClass]);
      }
      (*fMulticlassReturnVal).push_back(1.0/(1.0+norm));
   }


   
   return *fMulticlassReturnVal;
}






//_______________________________________________________________________
void TMVA::MethodDNN::AddWeightsXMLTo( void* parent ) const 
{
   // create XML description of DNN classifier
   // for all layers

   void* nn = gTools().xmlengine().NewChild(parent, 0, "Weights");
   void* xmlLayout = gTools().xmlengine().NewChild(nn, 0, "Layout");
   Int_t numLayers = fNet.layers ().size ();
   gTools().xmlengine().NewAttr(xmlLayout, 0, "NumberLayers", gTools().StringFromInt (numLayers) );
   for (Int_t i = 0; i < numLayers; i++) 
      {
         const TMVA::DNN::Layer& layer = fNet.layers ().at (i);
         int numNodes = layer.numNodes ();
         char activationFunction = (char)(layer.activationFunctionType ());
         int outputMode = (int)layer.modeOutputValues ();

         TString outputModeStr;
         outputModeStr.Form ("%d", outputMode);

         void* layerxml = gTools().xmlengine().NewChild(xmlLayout, 0, "Layer");
         gTools().xmlengine().NewAttr(layerxml, 0, "Connection",    TString("FULL") );
         gTools().xmlengine().NewAttr(layerxml, 0, "Nodes",    gTools().StringFromInt(numNodes) );
         gTools().xmlengine().NewAttr(layerxml, 0, "ActivationFunction",    TString (activationFunction) );
         gTools().xmlengine().NewAttr(layerxml, 0, "OutputMode",    outputModeStr);
      }


   void* weightsxml = gTools().xmlengine().NewChild(nn, 0, "Synapses");
   gTools().xmlengine().NewAttr (weightsxml, 0, "InputSize", gTools().StringFromInt((int)fNet.inputSize ()));
   gTools().xmlengine().NewAttr (weightsxml, 0, "OutputSize", gTools().StringFromInt((int)fNet.outputSize ()));
   gTools().xmlengine().NewAttr (weightsxml, 0, "NumberSynapses", gTools().StringFromInt((int)fWeights.size ()));
   std::stringstream s("");
   s.precision( 16 );
   for (std::vector<double>::const_iterator it = fWeights.begin (), itEnd = fWeights.end (); it != itEnd; ++it)
      {
         s << std::scientific << (*it) << " ";
      }
   gTools().xmlengine().AddRawLine (weightsxml, s.str().c_str());
}


//_______________________________________________________________________
void TMVA::MethodDNN::ReadWeightsFromXML( void* wghtnode )
{
   // read MLP from xml weight file
   fNet.clear ();

   void* nn = gTools().GetChild(wghtnode, "Weights");
   if (!nn)
      {
         //       std::cout << "no node \"Weights\" in XML, use weightnode" << std::endl;
         nn = wghtnode;
      }
   
   void* xmlLayout = NULL;
   xmlLayout = gTools().GetChild(wghtnode, "Layout");
   if (!xmlLayout)
      {
         std::cout << "no node Layout in XML" << std::endl;
         return;
      }


   
   //   std::cout << "read layout from XML" << std::endl;
   void* ch = gTools().xmlengine().GetChild (xmlLayout);
   TString connection;
   UInt_t numNodes;
   TString activationFunction;
   TString outputMode;
   fNet.clear ();
   while (ch) 
      {
         gTools().ReadAttr (ch, "Connection", connection);
         gTools().ReadAttr (ch, "Nodes", numNodes);
         gTools().ReadAttr (ch, "ActivationFunction", activationFunction);
         gTools().ReadAttr (ch, "OutputMode", outputMode);
         ch = gTools().GetNextChild(ch);

         fNet.addLayer (DNN::Layer (numNodes, (TMVA::DNN::EnumFunction)activationFunction (0), (DNN::ModeOutputValues)outputMode.Atoi ()));
      }

   //   std::cout << "read weights XML" << std::endl;

   void* xmlWeights  = NULL;
   xmlWeights = gTools().GetChild(wghtnode, "Synapses");
   if (!xmlWeights)
      return;

   Int_t numWeights (0);
   Int_t inputSize (0);
   Int_t outputSize (0);
   gTools().ReadAttr (xmlWeights, "NumberSynapses", numWeights);
   gTools().ReadAttr (xmlWeights, "InputSize", inputSize);
   gTools().ReadAttr (xmlWeights, "OutputSize", outputSize);
   fNet.setInputSize (inputSize);
   fNet.setOutputSize (outputSize); // num vars + bias node

   const char* content = gTools().GetContent (xmlWeights);
   std::stringstream sstr (content);
   for (Int_t iWeight = 0; iWeight<numWeights; ++iWeight) 
      { // synapses
         Double_t weight;
         sstr >> weight;
         fWeights.push_back (weight);
      }
}


//_______________________________________________________________________
void TMVA::MethodDNN::ReadWeightsFromStream( std::istream & /*istr*/)
{
   // // destroy/clear the network then read it back in from the weights file

   // // delete network so we can reconstruct network from scratch

   // TString dummy;

   // // synapse weights
   // Double_t weight;
   // std::vector<Double_t>* weights = new std::vector<Double_t>();
   // istr>> dummy;
   // while (istr>> dummy >> weight) weights->push_back(weight); // use w/ slower write-out

   // ForceWeights(weights);
   

   // delete weights;
}

//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodDNN::CreateRanking()
{
   // compute ranking of input variables by summing function of weights

   // create the ranking object
   fRanking = new Ranking( GetName(), "Importance" );

   for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {
      fRanking->AddRank( Rank( GetInputLabel(ivar), 1.0));
   }

   // TNeuron*  neuron;
   // TSynapse* synapse;
   // Double_t  importance, avgVal;
   // TString varName;

   // for (UInt_t ivar = 0; ivar < GetNvar(); ivar++) {

   //    neuron = GetInputNeuron(ivar);
   //    Int_t numSynapses = neuron->NumPostLinks();
   //    importance = 0;
   //    varName = GetInputVar(ivar); // fix this line

   //    // figure out average value of variable i
   //    Double_t meanS, meanB, rmsS, rmsB, xmin, xmax;
   //    Statistics( TMVA::Types::kTraining, varName, 
   //                meanS, meanB, rmsS, rmsB, xmin, xmax );

   //    avgVal = (TMath::Abs(meanS) + TMath::Abs(meanB))/2.0;
   //    double meanrms = (TMath::Abs(rmsS) + TMath::Abs(rmsB))/2.;
   //    if (avgVal<meanrms) avgVal = meanrms;      
   //    if (IsNormalised()) avgVal = 0.5*(1 + gTools().NormVariable( avgVal, GetXmin( ivar ), GetXmax( ivar ))); 

   //    for (Int_t j = 0; j < numSynapses; j++) {
   //       synapse = neuron->PostLinkAt(j);
   //       importance += synapse->GetWeight() * synapse->GetWeight();
   //    }
      
   //    importance *= avgVal * avgVal;

   //    fRanking->AddRank( Rank( varName, importance ) );
   // }

   return fRanking;
}






//_______________________________________________________________________
void TMVA::MethodDNN::MakeClassSpecific( std::ostream& /*fout*/, const TString& /*className*/ ) const
{
   // write specific classifier response
   //   MethodADNNBase::MakeClassSpecific(fout, className);
}

//_______________________________________________________________________
void TMVA::MethodDNN::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line:
   //         "|--------------------------------------------------------------|"
   TString col    = gConfig().WriteOptionsReference() ? TString() : gTools().Color("bold");
   TString colres = gConfig().WriteOptionsReference() ? TString() : gTools().Color("reset");

   Log() << Endl;
   Log() << col << "--- Short description:" << colres << Endl;
   Log() << Endl;
   Log() << "The DNN neural network is a feedforward" << Endl;
   Log() << "multilayer perceptron impementation. The DNN has a user-" << Endl;
   Log() << "defined hidden layer architecture, where the number of input (output)" << Endl;
   Log() << "nodes is determined by the input variables (output classes, i.e., " << Endl;
   Log() << "signal and one background, regression or multiclass). " << Endl;
   Log() << Endl;
   Log() << col << "--- Performance optimisation:" << colres << Endl;
   Log() << Endl;

   const char* txt = "The DNN supports various options to improve performance in terms of training speed and \n \
reduction of overfitting: \n \
\n \
      - different training settings can be stacked. Such that the initial training  \n\
        is done with a large learning rate and a large drop out fraction whilst \n \
        in a later stage learning rate and drop out can be reduced. \n \
      - drop out  \n \
        [recommended: \n \
         initial training stage: 0.0 for the first layer, 0.5 for later layers. \n \
         later training stage: 0.1 or 0.0 for all layers \n \
         final training stage: 0.0] \n \
        Drop out is a technique where a at each training cycle a fraction of arbitrary  \n \
        nodes is disabled. This reduces co-adaptation of weights and thus reduces overfitting. \n \
      - L1 and L2 regularization are available \n \
      - Minibatches  \n \
        [recommended 10 - 150] \n \
        Arbitrary mini-batch sizes can be chosen. \n \
      - Multithreading \n \
        [recommended: True] \n \
        Multithreading can be turned on. The minibatches are distributed to the available \n \
        cores. The algorithm is lock-free (\"Hogwild!\"-style) for each cycle. \n \
 \n \
      Options: \n \
      \"Layout\": \n \
          - example: \"TANH|(N+30)*2,TANH|(N+30),LINEAR\" \n \
          - meaning:  \n \
              . two hidden layers (separated by \",\") \n \
              . the activation function is TANH (other options: RELU, SOFTSIGN, LINEAR) \n \
              . the activation function for the output layer is LINEAR \n \
              . the first hidden layer has (N+30)*2 nodes where N is the number of input neurons \n \
              . the second hidden layer has N+30 nodes, where N is the number of input neurons \n \
              . the number of nodes in the output layer is determined by the number of output nodes \n \
                and can therefore not be chosen freely.  \n \
 \n \
       \"ErrorStrategy\": \n \
           - SUMOFSQUARES \n \
             The error of the neural net is determined by a sum-of-squares error function \n \
             For regression, this is the only possible choice.  \n \
           - CROSSENTROPY \n \
             The error of the neural net is determined by a cross entropy function. The \n \
             output values are automatically (internally) transformed into probabilities \n \
             using a sigmoid function. \n \
             For signal/background classification this is the default choice.  \n \
             For multiclass using cross entropy more than one or no output classes  \n \
             can be equally true or false (e.g. Event 0: A and B are true, Event 1:  \n \
             A and C is true, Event 2: C is true, ...) \n \
           - MUTUALEXCLUSIVE \n \
             In multiclass settings, exactly one of the output classes can be true (e.g. either A or B or C) \n \
 \n \
        \"WeightInitialization\" \n \
           - XAVIER \n \
             [recommended] \n \
             \"Xavier Glorot & Yoshua Bengio\"-style of initializing the weights. The weights are chosen randomly \n \
             such that the variance of the values of the nodes is preserved for each layer.  \n \
           - XAVIERUNIFORM \n \
             The same as XAVIER, but with uniformly distributed weights instead of gaussian weights \n \
           - LAYERSIZE \n \
             Random values scaled by the layer size \n \
 \n \
         \"TrainingStrategy\" \n \
           - example: \"LearningRate=1e-1,Momentum=0.3,Repetitions=3,ConvergenceSteps=50,BatchSize=30,TestRepetitions=7,WeightDecay=0.0,Renormalize=L2,DropConfig=0.0,DropRepetitions=5|LearningRate=1e-4,Momentum=0.3,Repetitions=3,ConvergenceSteps=50,BatchSize=20,TestRepetitions=7,WeightDecay=0.001,Renormalize=L2,DropFraction=0.0,DropRepetitions=5\" \n \
           - explanation: two stacked training settings separated by \"|\" \n \
             . first training setting: \"LearningRate=1e-1,Momentum=0.3,Repetitions=3,ConvergenceSteps=50,BatchSize=30,TestRepetitions=7,WeightDecay=0.0,Renormalize=L2,DropConfig=0.0,DropRepetitions=5\" \n \
             . second training setting : \"LearningRate=1e-4,Momentum=0.3,Repetitions=3,ConvergenceSteps=50,BatchSize=20,TestRepetitions=7,WeightDecay=0.001,Renormalize=L2,DropFractions=0.0,DropRepetitions=5\" \n \
             . LearningRate :  \n \
               - recommended for classification: 0.1 initially, 1e-4 later \n \
               - recommended for regression: 1e-4 and less \n \
             . Momentum : \n \
               preserve a fraction of the momentum for the next training batch [fraction = 0.0 - 1.0] \n \
             . Repetitions : \n \
               train \"Repetitions\" repetitions with the same minibatch before switching to the next one \n \
             . ConvergenceSteps :  \n \
               Assume that convergence is reached after \"ConvergenceSteps\" cycles where no improvement \n \
               of the error on the test samples has been found. (Mind that only at each \"TestRepetitions\"  \n \
               cycle the test sampes are evaluated and thus the convergence is checked) \n \
             . BatchSize \n \
               Size of the mini-batches.  \n \
             . TestRepetitions \n \
               Perform testing the neural net on the test samples each \"TestRepetitions\" cycle \n \
             . WeightDecay \n \
               If \"Renormalize\" is set to L1 or L2, \"WeightDecay\" provides the renormalization factor \n \
             . Renormalize \n \
               NONE, L1 (|w|) or L2 (w^2) \n \
             . DropConfig \n \
               Drop a fraction of arbitrary nodes of each of the layers according to the values given \n \
               in the DropConfig.  \n \
               [example: DropConfig=0.0+0.5+0.3 \n \
                meaning: drop no nodes in layer 0 (input layer), half of the nodes in layer 1 and 30% of the nodes \n \
                in layer 2 \n \
                recommended: leave all the nodes turned on for the input layer (layer 0) \n \
                turn off half of the nodes in later layers for the initial training; leave all nodes \n \
                turned on (0.0) in later training stages] \n \
             . DropRepetitions \n \
               Each \"DropRepetitions\" cycle the configuration of which nodes are dropped is changed \n \
               [recommended : 1] \n \
             . Multithreading \n \
               turn on multithreading [recommended: True] \n \
               \n";
          
   Log () << txt << Endl;
   
}



//_______________________________________________________________________
void  TMVA::MethodDNN::WriteMonitoringHistosToFile( void ) const
{
   // write histograms and PDFs to file for monitoring purposes

   Log() << kINFO << "Write monitoring histograms to file: " << BaseDir()->GetPath() << Endl;
   BaseDir()->cd();
}




void TMVA::MethodDNN::checkGradients ()
{
   size_t inputSize = 1;
   size_t outputSize = 1;

   fNet.clear ();

   fNet.setInputSize (inputSize);
   fNet.setOutputSize (outputSize);
   fNet.addLayer (DNN::Layer (100, DNN::EnumFunction::SOFTSIGN)); 
   fNet.addLayer (DNN::Layer (30, DNN::EnumFunction::SOFTSIGN)); 
   fNet.addLayer (DNN::Layer (outputSize, DNN::EnumFunction::LINEAR, DNN::ModeOutputValues::SIGMOID)); 
   fNet.setErrorFunction (DNN::ModeErrorFunction::CROSSENTROPY);
   //    net.setErrorFunction (ModeErrorFunction::SUMOFSQUARES);

   size_t numWeights = fNet.numWeights (inputSize);
   std::vector<double> weights (numWeights);
   //weights.at (0) = 1000213.2;

   std::vector<Pattern> pattern;
   for (size_t iPat = 0, iPatEnd = 10; iPat < iPatEnd; ++iPat)
      {
         std::vector<double> input;
         std::vector<double> output;
         for (size_t i = 0; i < inputSize; ++i)
            {
               input.push_back (TMVA::DNN::gaussDouble (0.1, 4));
            }
         for (size_t i = 0; i < outputSize; ++i)
            {
               output.push_back (TMVA::DNN::gaussDouble (0, 3));
            }
         pattern.push_back (Pattern (input,output));
      }


   DNN::Settings settings (TString ("checkGradients"), /*_convergenceSteps*/ 15, /*_batchSize*/ 1, /*_testRepetitions*/ 7, /*_factorWeightDecay*/ 0, /*regularization*/ TMVA::DNN::EnumRegularization::NONE);

   size_t improvements = 0;
   size_t worsenings = 0;
   size_t smallDifferences = 0;
   size_t largeDifferences = 0;
   for (size_t iTest = 0; iTest < 1000; ++iTest)
      {
         TMVA::DNN::uniformDouble (weights, 0.7);
         std::vector<double> gradients (numWeights, 0);
         DNN::Batch batch (begin (pattern), end (pattern));
         DNN::DropContainer dropContainer;
         std::tuple<DNN::Settings&, DNN::Batch&, DNN::DropContainer&> settingsAndBatch (settings, batch, dropContainer);
         double E = fNet (settingsAndBatch, weights, gradients);
         std::vector<double> changedWeights;
         changedWeights.assign (weights.begin (), weights.end ());

         int changeWeightPosition = TMVA::DNN::randomInt (numWeights);
         double dEdw = gradients.at (changeWeightPosition);
         while (dEdw == 0.0)
            {
               changeWeightPosition = TMVA::DNN::randomInt (numWeights);
               dEdw = gradients.at (changeWeightPosition);
            }

         const double gamma = 0.01;
         double delta = gamma*dEdw;
         changedWeights.at (changeWeightPosition) += delta;
         if (dEdw == 0.0)
            {
               std::cout << "dEdw == 0.0 ";
               continue;
            }
        
         assert (dEdw != 0.0);
         double Echanged = fNet (settingsAndBatch, changedWeights);

         //       double difference = fabs((E-Echanged) - delta*dEdw);
         double difference = fabs ((E+delta - Echanged)/E);
         bool direction = (E-Echanged)>0 ? true : false;
         //       bool directionGrad = delta>0 ? true : false;
         bool isOk = difference < 0.3 && difference != 0;

         if (direction)
            ++improvements;
         else
            ++worsenings;

         if (isOk)
            ++smallDifferences;
         else
            ++largeDifferences;

         if (true || !isOk)
            {
               if (!direction)
                  std::cout << "=================" << std::endl;
               std::cout << "E = " << E << " Echanged = " << Echanged << " delta = " << delta << "   pos=" << changeWeightPosition << "   dEdw=" << dEdw << "  difference= " << difference << "  dirE= " << direction << std::endl;
            }
         if (isOk)
            {
            }
         else
            {
               //            for_each (begin (weights), end (weights), [](double w){ std::cout << w << ", "; });
               //            std::cout << std::endl;
               //            assert (isOk);
            }
      }
   std::cout << "improvements = " << improvements << std::endl;
   std::cout << "worsenings = " << worsenings << std::endl;
   std::cout << "smallDifferences = " << smallDifferences << std::endl;
   std::cout << "largeDifferences = " << largeDifferences << std::endl;

   std::cout << "check gradients done" << std::endl;
}

