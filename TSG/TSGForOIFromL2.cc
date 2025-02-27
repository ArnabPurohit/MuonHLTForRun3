/**
  \class    TSGForOIFromL2
  \brief    Create L3MuonTrajectorySeeds from L2 Muons updated at vertex in an outside-in manner
  \author   Benjamin Radburn-Smith, Santiago Folgueras, Bibhuprasad Mahakud, Jan Frederik Schulte, Dmitry Kondratyev (Purdue University, West Lafayette, USA)
 */

#include "RecoMuon/TrackerSeedGenerator/plugins/TSGForOIFromL2.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <memory>

TSGForOIFromL2::TSGForOIFromL2(const edm::ParameterSet& iConfig)
    : src_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      maxSeeds_(iConfig.getParameter<uint32_t>("maxSeeds")),
      maxHitSeeds_(iConfig.getParameter<uint32_t>("maxHitSeeds")),
      maxHitlessSeeds_(iConfig.getParameter<uint32_t>("maxHitlessSeeds")),
      numOfLayersToTry_(iConfig.getParameter<int32_t>("layersToTry")),
      numOfHitsToTry_(iConfig.getParameter<int32_t>("hitsToTry")),
      numL2ValidHitsCutAllEta_(iConfig.getParameter<uint32_t>("numL2ValidHitsCutAllEta")),
      numL2ValidHitsCutAllEndcap_(iConfig.getParameter<uint32_t>("numL2ValidHitsCutAllEndcap")),
      fixedErrorRescalingForHits_(iConfig.getParameter<double>("fixedErrorRescaleFactorForHits")),
      fixedErrorRescalingForHitless_(iConfig.getParameter<double>("fixedErrorRescaleFactorForHitless")),
      adjustErrorsDynamicallyForHits_(iConfig.getParameter<bool>("adjustErrorsDynamicallyForHits")),
      adjustErrorsDynamicallyForHitless_(iConfig.getParameter<bool>("adjustErrorsDynamicallyForHitless")),
      estimatorName_(iConfig.getParameter<std::string>("estimator")),
      minEtaForTEC_(iConfig.getParameter<double>("minEtaForTEC")),
      maxEtaForTOB_(iConfig.getParameter<double>("maxEtaForTOB")),
      useHitLessSeeds_(iConfig.getParameter<bool>("UseHitLessSeeds")),
      updator_(new KFUpdator()),
      measurementTrackerTag_(
          consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>("MeasurementTrackerEvent"))),
      pT1_(iConfig.getParameter<double>("pT1")),
      pT2_(iConfig.getParameter<double>("pT2")),
      pT3_(iConfig.getParameter<double>("pT3")),
      eta1_(iConfig.getParameter<double>("eta1")),
      eta2_(iConfig.getParameter<double>("eta2")),
      eta3_(iConfig.getParameter<double>("eta3")),
      eta4_(iConfig.getParameter<double>("eta4")),
      eta5_(iConfig.getParameter<double>("eta5")),
      eta6_(iConfig.getParameter<double>("eta6")),
      eta7_(iConfig.getParameter<double>("eta7")),
      SF1_(iConfig.getParameter<double>("SF1")),
      SF2_(iConfig.getParameter<double>("SF2")),
      SF3_(iConfig.getParameter<double>("SF3")),
      SF4_(iConfig.getParameter<double>("SF4")),
      SF5_(iConfig.getParameter<double>("SF5")),
      SF6_(iConfig.getParameter<double>("SF6")),
      tsosDiff1_(iConfig.getParameter<double>("tsosDiff1")),
      tsosDiff2_(iConfig.getParameter<double>("tsosDiff2")),
      propagatorName_(iConfig.getParameter<std::string>("propagatorName")),
      theCategory_(std::string("Muon|RecoMuon|TSGForOIFromL2")),
      useBothAsInRun2_(iConfig.getParameter<bool>("useBothAsInRun2")),
      dontCreateHitbasedInBarrelAsInRun2_(iConfig.getParameter<bool>("dontCreateHitbasedInBarrelAsInRun2")),
      maxHitlessSeedsIP_(iConfig.getParameter<uint32_t>("maxHitlessSeedsIP")),
      maxHitlessSeedsMuS_(iConfig.getParameter<uint32_t>("maxHitlessSeedsMuS")),
      maxHitDoubletSeeds_(iConfig.getParameter<uint32_t>("maxHitDoubletSeeds")),
      getStrategyFromDNN_(iConfig.getParameter<bool>("getStrategyFromDNN")),
      etaSplitForDnn_(iConfig.getParameter<double>("etaSplitForDnn")),
      dnnModelPath_barrel_(iConfig.getParameter<std::string>("dnnModelPath_barrel")),
      dnnMetadataPath_barrel_(iConfig.getParameter<std::string>("dnnMetadataPath_barrel")),
      dnnModelPath_endcap_(iConfig.getParameter<std::string>("dnnModelPath_endcap")),
      dnnMetadataPath_endcap_(iConfig.getParameter<std::string>("dnnMetadataPath_endcap"))
{
  if (getStrategyFromDNN_){
      tensorflow::setLogging("2");

      edm::FileInPath dnnPath_barrel(dnnModelPath_barrel_);
      graphDef_barrel_ = tensorflow::loadGraphDef(dnnPath_barrel.fullPath());
      tf_session_barrel_ = tensorflow::createSession(graphDef_barrel_);
      edm::FileInPath dnnMetadataPath_barrel_full(dnnMetadataPath_barrel_);
      metadataFile_barrel_ = TFile::Open(dnnMetadataPath_barrel_full.fullPath().c_str());
      inpOrderHist_barrel_ = (TH1D*)(metadataFile_barrel_->Get("input_order"));
      layerNamesHist_barrel_ = (TH1D*)(metadataFile_barrel_->Get("layer_names"));
      decoderHist_barrel_ = (TH2D*)(metadataFile_barrel_->Get("scheme"));

      edm::FileInPath dnnPath_endcap(dnnModelPath_endcap_);
      graphDef_endcap_ = tensorflow::loadGraphDef(dnnPath_endcap.fullPath());
      tf_session_endcap_ = tensorflow::createSession(graphDef_endcap_);
      edm::FileInPath dnnMetadataPath_endcap_full(dnnMetadataPath_endcap_);
      metadataFile_endcap_ = TFile::Open(dnnMetadataPath_endcap_full.fullPath().c_str());
      inpOrderHist_endcap_ = (TH1D*)(metadataFile_endcap_->Get("input_order"));
      layerNamesHist_endcap_ = (TH1D*)(metadataFile_endcap_->Get("layer_names"));
      decoderHist_endcap_ = (TH2D*)(metadataFile_endcap_->Get("scheme"));
  }
  produces<std::vector<TrajectorySeed> >();
}

TSGForOIFromL2::~TSGForOIFromL2() {
    if (getStrategyFromDNN_){
        tensorflow::closeSession(tf_session_barrel_);
        tensorflow::closeSession(tf_session_endcap_);
        delete graphDef_barrel_;
        delete graphDef_endcap_;
        metadataFile_barrel_->Close();
        metadataFile_endcap_->Close();
    }
}

//
// Produce seeds
//
void TSGForOIFromL2::produce(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Initialize variables
  unsigned int numSeedsMade = 0;
  unsigned int layerCount = 0;
  unsigned int hitlessSeedsMadeIP = 0;
  unsigned int hitlessSeedsMadeMuS = 0;
  unsigned int hitSeedsMade = 0;
  unsigned int hitDoubletSeedsMade = 0;

  // Surface used to make a TSOS at the PCA to the beamline
  Plane::PlanePointer dummyPlane = Plane::build(Plane::PositionType(), Plane::RotationType());

  // Read ESHandles
  edm::Handle<MeasurementTrackerEvent> measurementTrackerH;
  edm::ESHandle<Chi2MeasurementEstimatorBase> estimatorH;
  edm::ESHandle<MagneticField> magfieldH;
  edm::ESHandle<Propagator> propagatorAlongH;
  edm::ESHandle<Propagator> propagatorOppositeH;
  edm::ESHandle<TrackerGeometry> tmpTkGeometryH;
  edm::ESHandle<GlobalTrackingGeometry> geometryH;
  edm::ESHandle<NavigationSchool> navSchool;

  iSetup.get<IdealMagneticFieldRecord>().get(magfieldH);
  iSetup.get<TrackingComponentsRecord>().get(propagatorName_, propagatorOppositeH);
  iSetup.get<TrackingComponentsRecord>().get(propagatorName_, propagatorAlongH);
  iSetup.get<GlobalTrackingGeometryRecord>().get(geometryH);
  iSetup.get<TrackerDigiGeometryRecord>().get(tmpTkGeometryH);
  iSetup.get<TrackingComponentsRecord>().get(estimatorName_, estimatorH);
  iEvent.getByToken(measurementTrackerTag_, measurementTrackerH);
  iSetup.get<NavigationSchoolRecord>().get("SimpleNavigationSchool", navSchool);

  // Read L2 track collection
  edm::Handle<reco::TrackCollection> l2TrackCol;
  iEvent.getByToken(src_, l2TrackCol);

  // The product
  std::unique_ptr<std::vector<TrajectorySeed> > result(new std::vector<TrajectorySeed>());

  // Get vector of Detector layers
  std::vector<BarrelDetLayer const*> const& tob = measurementTrackerH->geometricSearchTracker()->tobLayers();
  std::vector<ForwardDetLayer const*> const& tecPositive =
      tmpTkGeometryH->isThere(GeomDetEnumerators::P2OTEC)
          ? measurementTrackerH->geometricSearchTracker()->posTidLayers()
          : measurementTrackerH->geometricSearchTracker()->posTecLayers();
  std::vector<ForwardDetLayer const*> const& tecNegative =
      tmpTkGeometryH->isThere(GeomDetEnumerators::P2OTEC)
          ? measurementTrackerH->geometricSearchTracker()->negTidLayers()
          : measurementTrackerH->geometricSearchTracker()->negTecLayers();

  // Get suitable propagators
  std::unique_ptr<Propagator> propagatorAlong = SetPropagationDirection(*propagatorAlongH, alongMomentum);
  std::unique_ptr<Propagator> propagatorOpposite = SetPropagationDirection(*propagatorOppositeH, oppositeToMomentum);

  // Stepping Helix Propagator for propogation from muon system to tracker
  edm::ESHandle<Propagator> SHPOpposite;
  iSetup.get<TrackingComponentsRecord>().get("hltESPSteppingHelixPropagatorOpposite", SHPOpposite);

  // Loop over the L2's and make seeds for all of them
  LogTrace(theCategory_) << "TSGForOIFromL2::produce: Number of L2's: " << l2TrackCol->size();
  for (unsigned int l2TrackColIndex(0); l2TrackColIndex != l2TrackCol->size(); ++l2TrackColIndex) {
    const reco::TrackRef l2(l2TrackCol, l2TrackColIndex);

    // Container of Seeds
    std::vector<TrajectorySeed> out;
    LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::produce: L2 muon pT, eta, phi --> " << l2->pt() << " , " << l2->eta()
                               << " , " << l2->phi() << std::endl;

    FreeTrajectoryState fts = trajectoryStateTransform::initialFreeState(*l2, magfieldH.product());

    dummyPlane->move(fts.position() - dummyPlane->position());
    TrajectoryStateOnSurface tsosAtIP = TrajectoryStateOnSurface(fts, *dummyPlane);
    LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::produce: Created TSOSatIP: " << tsosAtIP << std::endl;

    // Get the TSOS on the innermost layer of the L2
    TrajectoryStateOnSurface tsosAtMuonSystem =
        trajectoryStateTransform::innerStateOnSurface(*l2, *geometryH, magfieldH.product());
    LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::produce: Created TSOSatMuonSystem: " << tsosAtMuonSystem
                               << std::endl;

    LogTrace("TSGForOIFromL2")
        << "TSGForOIFromL2::produce: Check the error of the L2 parameter and use hit seeds if big errors" << std::endl;

    StateOnTrackerBound fromInside(propagatorAlong.get());
    TrajectoryStateOnSurface outerTkStateInside = fromInside(fts);

    StateOnTrackerBound fromOutside(&*SHPOpposite);
    TrajectoryStateOnSurface outerTkStateOutside = fromOutside(tsosAtMuonSystem);

    // Check if the two positions (using updated and not-updated TSOS) agree withing certain extent.
    // If both TSOSs agree, use only the one at vertex, as it uses more information. If they do not agree, search for seeds based on both.
    double L2muonEta = l2->eta();
    double absL2muonEta = std::abs(L2muonEta);
    bool useBoth = false;
    
    // make non-const copies of parameters
    // (we want to override them if DNN evaluation is enabled)
    unsigned int maxHitSeeds__ = maxHitSeeds_;
    unsigned int maxHitDoubletSeeds__ = maxHitDoubletSeeds_;
    unsigned int maxHitlessSeedsIP__ = maxHitlessSeedsIP_;
    unsigned int maxHitlessSeedsMuS__ = maxHitlessSeedsMuS_; 
    bool dontCreateHitbasedInBarrelAsInRun2__ = dontCreateHitbasedInBarrelAsInRun2_;
    bool useBothAsInRun2__ = useBothAsInRun2_;
    
    // update strategy parameters by evaluating DNN
    if (getStrategyFromDNN_){
        int nHBd(0), nHLIP(0), nHLMuS(0);
        bool dnnSuccess_ = false;

        // Put variables needed for DNN into an std::map
        std::map<std::string, float> feature_map_ = getFeatureMap(l2, tsosAtIP, outerTkStateOutside);
        //for (auto const &pair: feature_map_) {
        //    std::cout << pair.first << " = " << pair.second << srd::endl;
        //}

        if (std::abs(l2->eta())<etaSplitForDnn_){
            // barrel
            evaluateDnn(
                feature_map_, tf_session_barrel_,
                inpOrderHist_barrel_, layerNamesHist_barrel_, decoderHist_barrel_,
                nHBd, nHLIP, nHLMuS, dnnSuccess_
            );
        } else {
            // endcap
            evaluateDnn(
                feature_map_, tf_session_endcap_,
                inpOrderHist_endcap_, layerNamesHist_endcap_, decoderHist_endcap_,
                nHBd, nHLIP, nHLMuS, dnnSuccess_
            );
        }
        if (!dnnSuccess_) break;
        //std::cout << "DNN decision: " << nHBd << " " << nHLIP << " " << nHLMuS << std::endl;
        maxHitSeeds__ = 0;
        maxHitDoubletSeeds__ = nHBd;
        maxHitlessSeedsIP__ = nHLIP;
        maxHitlessSeedsMuS__ = nHLMuS;
        
        dontCreateHitbasedInBarrelAsInRun2__ = false;
        useBothAsInRun2__ = false;
    }

    if (useBothAsInRun2__ && outerTkStateInside.isValid() && outerTkStateOutside.isValid()) {
      if (l2->numberOfValidHits() < numL2ValidHitsCutAllEta_)
        useBoth = true;
      if (l2->numberOfValidHits() < numL2ValidHitsCutAllEndcap_ && absL2muonEta > eta7_)
        useBoth = true;
      if (absL2muonEta > eta1_ && absL2muonEta < eta1_)
        useBoth = true;
    }
    
    numSeedsMade = 0;
    hitlessSeedsMadeIP = 0;
    hitlessSeedsMadeMuS = 0;
    hitSeedsMade = 0;
    hitDoubletSeedsMade = 0;

    // calculate scale factors
    double errorSFHits = (adjustErrorsDynamicallyForHits_ ? calculateSFFromL2(l2) : fixedErrorRescalingForHits_);
    double errorSFHitless =
        (adjustErrorsDynamicallyForHitless_ ? calculateSFFromL2(l2) : fixedErrorRescalingForHitless_);

    // BARREL
    if (absL2muonEta < maxEtaForTOB_) {
      layerCount = 0;
      for (auto it = tob.rbegin(); it != tob.rend(); ++it) {
        LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::produce: looping in TOB layer " << layerCount << std::endl;
        if (useHitLessSeeds_ && hitlessSeedsMadeIP < maxHitlessSeedsIP__ && numSeedsMade < maxSeeds_)
          makeSeedsWithoutHits(**it,
                               tsosAtIP,
                               *(propagatorAlong.get()),
                               estimatorH,
                               errorSFHitless,
                               hitlessSeedsMadeIP,
                               numSeedsMade,
                               out);
        if (outerTkStateInside.isValid() && outerTkStateOutside.isValid() &&
            useHitLessSeeds_ && hitlessSeedsMadeMuS < maxHitlessSeedsMuS__ && numSeedsMade < maxSeeds_)
            makeSeedsWithoutHits(**it,
                                 outerTkStateOutside,
                                 *(propagatorOpposite.get()),
                                 estimatorH,
                                 errorSFHitless,
                                 hitlessSeedsMadeMuS,
                                 numSeedsMade,
                                 out);
        // Do not create hitbased seeds in barrel region
        if (hitSeedsMade < maxHitSeeds__ && numSeedsMade < maxSeeds_){
            // Run2 approach, preserved for backward compatibility
            if (!(dontCreateHitbasedInBarrelAsInRun2__ && (absL2muonEta <= 1.0)))
              makeSeedsFromHits(**it,
                            tsosAtIP,
                            *(propagatorAlong.get()),
                            estimatorH,
                            measurementTrackerH,
                            errorSFHits,
                            hitSeedsMade,
                            numSeedsMade,
                            layerCount,
                            out);
        }

        if (hitDoubletSeedsMade < maxHitDoubletSeeds__ && numSeedsMade < maxSeeds_){
            makeSeedsFromHitDoublets(**it,
                            tsosAtIP,
                            *(propagatorAlong.get()),
                            estimatorH,
                            measurementTrackerH,
                            navSchool,
                            errorSFHits,
                            hitDoubletSeedsMade,
                            numSeedsMade,
                            layerCount,
                            out);
        }
        // Run2 approach, preserved for backward compatibility
        if (useBoth) {
          if (useHitLessSeeds_ && hitlessSeedsMadeMuS < maxHitlessSeedsIP__ && numSeedsMade < maxSeeds_)
            makeSeedsWithoutHits(**it,
                                 outerTkStateOutside,
                                 *(propagatorOpposite.get()),
                                 estimatorH,
                                 errorSFHitless,
                                 hitlessSeedsMadeMuS,
                                 numSeedsMade,
                                 out);
        }
      }
      LogTrace("TSGForOIFromL2") << "TSGForOIFromL2:::produce: NumSeedsMade = " << numSeedsMade
                                 << " , layerCount = " << layerCount << std::endl;
    }

    // Reset number of seeds if in overlap region
    if (absL2muonEta > minEtaForTEC_ && absL2muonEta < maxEtaForTOB_) {
      numSeedsMade = 0;
      hitlessSeedsMadeIP = 0;
      hitlessSeedsMadeMuS = 0;
      hitSeedsMade = 0;
      hitDoubletSeedsMade = 0;
    }

    // ENDCAP+
    if (L2muonEta > minEtaForTEC_) {
      layerCount = 0;
      for (auto it = tecPositive.rbegin(); it != tecPositive.rend(); ++it) {
        LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::produce: looping in TEC+ layer " << layerCount << std::endl;
        if (useHitLessSeeds_ && hitlessSeedsMadeIP < maxHitlessSeedsIP__ && numSeedsMade < maxSeeds_)
          makeSeedsWithoutHits(**it,
                               tsosAtIP,
                               *(propagatorAlong.get()),
                               estimatorH,
                               errorSFHitless,
                               hitlessSeedsMadeIP,
                               numSeedsMade,
                               out);
        if (outerTkStateInside.isValid() && outerTkStateOutside.isValid() &&
            useHitLessSeeds_ && hitlessSeedsMadeMuS < maxHitlessSeedsMuS__ && numSeedsMade < maxSeeds_)
            makeSeedsWithoutHits(**it,
                                 outerTkStateOutside,
                                 *(propagatorOpposite.get()),
                                 estimatorH,
                                 errorSFHitless,
                                 hitlessSeedsMadeMuS,
                                 numSeedsMade,
                                 out);
        if (hitSeedsMade < maxHitSeeds__ && numSeedsMade < maxSeeds_){
            // Run2 approach, preserved for backward compatibility
            if (!(dontCreateHitbasedInBarrelAsInRun2__ && (absL2muonEta <= 1.0)))
              makeSeedsFromHits(**it,
                            tsosAtIP,
                            *(propagatorAlong.get()),
                            estimatorH,
                            measurementTrackerH,
                            errorSFHits,
                            hitSeedsMade,
                            numSeedsMade,
                            layerCount,
                            out);
        }
        if (hitDoubletSeedsMade < maxHitDoubletSeeds__ && numSeedsMade < maxSeeds_){
            makeSeedsFromHitDoublets(**it,
                            tsosAtIP,
                            *(propagatorAlong.get()),
                            estimatorH,
                            measurementTrackerH,
                            navSchool,
                            errorSFHits,
                            hitDoubletSeedsMade,
                            numSeedsMade,
                            layerCount,
                            out);
        }
         // Run2 approach, preserved for backward compatibility
        if (useBoth) {
          if (useHitLessSeeds_ && hitlessSeedsMadeMuS < maxHitlessSeedsIP__ && numSeedsMade < maxSeeds_)
            makeSeedsWithoutHits(**it,
                                 outerTkStateOutside,
                                 *(propagatorOpposite.get()),
                                 estimatorH,
                                 errorSFHitless,
                                 hitlessSeedsMadeMuS,
                                 numSeedsMade,
                                 out);
        }
      }
      LogTrace("TSGForOIFromL2") << "TSGForOIFromL2:::produce: NumSeedsMade = " << numSeedsMade
                                 << " , layerCount = " << layerCount << std::endl;
    }

    // ENDCAP-
    if (L2muonEta < -minEtaForTEC_) {
      layerCount = 0;
      for (auto it = tecNegative.rbegin(); it != tecNegative.rend(); ++it) {
        LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::produce: looping in TEC- layer " << layerCount << std::endl;
        if (useHitLessSeeds_ && hitlessSeedsMadeIP < maxHitlessSeedsIP__ && numSeedsMade < maxSeeds_)
          makeSeedsWithoutHits(**it,
                               tsosAtIP,
                               *(propagatorAlong.get()),
                               estimatorH,
                               errorSFHitless,
                               hitlessSeedsMadeIP,
                               numSeedsMade,
                               out);
        if (outerTkStateInside.isValid() && outerTkStateOutside.isValid() &&
            useHitLessSeeds_ && hitlessSeedsMadeMuS < maxHitlessSeedsMuS__ && numSeedsMade < maxSeeds_)
            makeSeedsWithoutHits(**it,
                                 outerTkStateOutside,
                                 *(propagatorOpposite.get()),
                                 estimatorH,
                                 errorSFHitless,
                                 hitlessSeedsMadeMuS,
                                 numSeedsMade,
                                 out);

        if (hitSeedsMade < maxHitSeeds__ && numSeedsMade < maxSeeds_){
            // Run2 approach, preserved for backward compatibility
            if (!(dontCreateHitbasedInBarrelAsInRun2__ && (absL2muonEta <= 1.0)))
              makeSeedsFromHits(**it,
                            tsosAtIP,
                            *(propagatorAlong.get()),
                            estimatorH,
                            measurementTrackerH,
                            errorSFHits,
                            hitSeedsMade,
                            numSeedsMade,
                            layerCount,
                            out);
        }
        if (hitDoubletSeedsMade < maxHitDoubletSeeds__ && numSeedsMade < maxSeeds_){
            makeSeedsFromHitDoublets(**it,
                            tsosAtIP,
                            *(propagatorAlong.get()),
                            estimatorH,
                            measurementTrackerH,
                            navSchool,
                            errorSFHits,
                            hitDoubletSeedsMade,
                            numSeedsMade,
                            layerCount,
                            out);
        }
        // Run2 approach, preserved for backward compatibility
        if (useBoth) {
          if (useHitLessSeeds_ && hitlessSeedsMadeMuS < maxHitlessSeedsIP__ && numSeedsMade < maxSeeds_)
            makeSeedsWithoutHits(**it,
                                 outerTkStateOutside,
                                 *(propagatorOpposite.get()),
                                 estimatorH,
                                 errorSFHitless,
                                 hitlessSeedsMadeMuS,
                                 numSeedsMade,
                                 out);
        }
      }
      LogTrace("TSGForOIFromL2") << "TSGForOIFromL2:::produce: NumSeedsMade = " << numSeedsMade
                                 << " , layerCount = " << layerCount << std::endl;
    }

    for (std::vector<TrajectorySeed>::iterator it = out.begin(); it != out.end(); ++it) {
      result->push_back(*it);
    }

  }  // L2Collection

  edm::LogInfo(theCategory_) << "TSGForOIFromL2::produce: number of seeds made: " << result->size();

  iEvent.put(std::move(result));
}

//
// Create seeds without hits on a given layer (TOB or TEC)
//
void TSGForOIFromL2::makeSeedsWithoutHits(const GeometricSearchDet& layer,
                                          const TrajectoryStateOnSurface& tsos,
                                          const Propagator& propagatorAlong,
                                          edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
                                          double errorSF,
                                          unsigned int& hitlessSeedsMade,
                                          unsigned int& numSeedsMade,
                                          std::vector<TrajectorySeed>& out) const {
  // create hitless seeds
  LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::makeSeedsWithoutHits: Start hitless" << std::endl;
  std::vector<GeometricSearchDet::DetWithState> dets;
  layer.compatibleDetsV(tsos, propagatorAlong, *estimator, dets);
  if (!dets.empty()) {
    auto const& detOnLayer = dets.front().first;
    auto const& tsosOnLayer = dets.front().second;
    LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::makeSeedsWithoutHits: tsosOnLayer " << tsosOnLayer << std::endl;
    if (!tsosOnLayer.isValid()) {
      edm::LogInfo(theCategory_) << "ERROR!: Hitless TSOS is not valid!";
    } else {
      dets.front().second.rescaleError(errorSF);
      PTrajectoryStateOnDet const& ptsod =
          trajectoryStateTransform::persistentState(tsosOnLayer, detOnLayer->geographicalId().rawId());
      TrajectorySeed::RecHitContainer rHC;
      out.push_back(TrajectorySeed(ptsod, rHC, oppositeToMomentum));
      LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::makeSeedsWithoutHits: TSOS (Hitless) done " << std::endl;
      hitlessSeedsMade++;
      numSeedsMade++;
    }
  }
}

//
// Find hits on a given layer (TOB or TEC) and create seeds from updated TSOS with hit
//
void TSGForOIFromL2::makeSeedsFromHits(const GeometricSearchDet& layer,
                                       const TrajectoryStateOnSurface& tsos,
                                       const Propagator& propagatorAlong,
                                       edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
                                       edm::Handle<MeasurementTrackerEvent>& measurementTracker,
                                       double errorSF,
                                       unsigned int& hitSeedsMade,
                                       unsigned int& numSeedsMade,
                                       unsigned int& layerCount,
                                       std::vector<TrajectorySeed>& out) const {
  if (layerCount > numOfLayersToTry_)
    return;

  // Error Rescaling
  TrajectoryStateOnSurface onLayer(tsos);
  onLayer.rescaleError(errorSF);

  std::vector<GeometricSearchDet::DetWithState> dets;
  layer.compatibleDetsV(onLayer, propagatorAlong, *estimator, dets);

  // Find Measurements on each DetWithState
  LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::makeSeedsFromHits: Find measurements on each detWithState  "
                             << dets.size() << std::endl;
  std::vector<TrajectoryMeasurement> meas;
  for (std::vector<GeometricSearchDet::DetWithState>::iterator it = dets.begin(); it != dets.end(); ++it) {
    MeasurementDetWithData det = measurementTracker->idToDet(it->first->geographicalId());
    if (det.isNull())
      continue;
    if (!it->second.isValid())
      continue;  // Skip if TSOS is not valid

    std::vector<TrajectoryMeasurement> mymeas =
        det.fastMeasurements(it->second, onLayer, propagatorAlong, *estimator);  // Second TSOS is not used
    for (std::vector<TrajectoryMeasurement>::const_iterator it2 = mymeas.begin(), ed2 = mymeas.end(); it2 != ed2;
         ++it2) {
      if (it2->recHit()->isValid())
        meas.push_back(*it2);  // Only save those which are valid
    }
  }

  // Update TSOS using TMs after sorting, then create Trajectory Seed and put into vector
  LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::makeSeedsFromHits: Update TSOS using TMs after sorting, then create "
                                "Trajectory Seed, number of TM = "
                             << meas.size() << std::endl;
  std::sort(meas.begin(), meas.end(), TrajMeasLessEstim());

  unsigned int found = 0;
  for (std::vector<TrajectoryMeasurement>::const_iterator it = meas.begin(); it != meas.end(); ++it) {
    TrajectoryStateOnSurface updatedTSOS = updator_->update(it->forwardPredictedState(), *it->recHit());
    LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::makeSeedsFromHits: TSOS for TM " << found << std::endl;
    if (not updatedTSOS.isValid())
      continue;

    edm::OwnVector<TrackingRecHit> seedHits;
    seedHits.push_back(*it->recHit()->hit());
    PTrajectoryStateOnDet const& pstate =
        trajectoryStateTransform::persistentState(updatedTSOS, it->recHit()->geographicalId().rawId());
    LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::makeSeedsFromHits: Number of seedHits: " << seedHits.size()
                               << std::endl;
    TrajectorySeed seed(pstate, std::move(seedHits), oppositeToMomentum);
    out.push_back(seed);
    found++;
    numSeedsMade++;
    hitSeedsMade++;
    if (found == numOfHitsToTry_)
      break;
    if (hitSeedsMade > maxHitSeeds_)
      return;
  }

  if (found)
    layerCount++;
}



void TSGForOIFromL2::makeSeedsFromHitDoublets(const GeometricSearchDet& layer,
                                       const TrajectoryStateOnSurface& tsos,
                                       const Propagator& propagatorAlong,
                                       edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
                                       edm::Handle<MeasurementTrackerEvent>& measurementTracker,
                                       edm::ESHandle<NavigationSchool> navSchool,
                                       double errorSF,
                                       unsigned int& hitDoubletSeedsMade,
                                       unsigned int& numSeedsMade,
                                       unsigned int& layerCount,
                                       std::vector<TrajectorySeed>& out) const {

  // This method is similar to makeSeedsFromHits, but the seed is created
  // only when in addition to a hit on a given layer, there are more compatible hits
  // on next layers (going from outside inwards), compatible with updated TSOS.
  // If that's the case, multiple compatible hits are used to create a single seed.

  // Configured to only check the immideately adjacent layer and add one more hit
  int max_addtnl_layers = 1; // max number of additional layers to scan
  int max_meas = 1; // number of measurements to consider on each additional layer

  // // // First, regular procedure to find a compatible hit - like in makeSeedsFromHits // // //
  
  // Error Rescaling
  TrajectoryStateOnSurface onLayer(tsos);
  onLayer.rescaleError(errorSF);

  // Find dets compatible with original TSOS
  std::vector< GeometricSearchDet::DetWithState > dets;
  layer.compatibleDetsV(onLayer, propagatorAlong, *estimator, dets);

  LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::makeSeedsFromHitDoublets: Find measurements on each detWithState  " << dets.size() << std::endl;
  std::vector<TrajectoryMeasurement> meas;
    
  // Loop over dets
  for (std::vector<GeometricSearchDet::DetWithState>::iterator idet=dets.begin(); idet!=dets.end(); ++idet) {
    MeasurementDetWithData det = measurementTracker->idToDet(idet->first->geographicalId());

    if (det.isNull()) continue;    // skip if det does not exist
    if (!idet->second.isValid()) continue;    // skip if TSOS is invalid

    // Find measurements on this det
    std::vector <TrajectoryMeasurement> mymeas = det.fastMeasurements(idet->second, onLayer, propagatorAlong, *estimator);
    
    // Save valid measurements 
    for (std::vector<TrajectoryMeasurement>::const_iterator imea = mymeas.begin(), ed2 = mymeas.end(); imea != ed2; ++imea) {
      if (imea->recHit()->isValid()) meas.push_back(*imea);
    } // end loop over meas
  } // end loop over dets

  LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::makeSeedsFromHitDoublets: Update TSOS using TMs after sorting, then create Trajectory Seed, number of TM = " << meas.size() << std::endl;
  
  // sort valid measurements found on the first layer
  std::sort(meas.begin(), meas.end(), TrajMeasLessEstim());

  unsigned int found = 0;
  int hit_num = 0;
  
  // Loop over all valid measurements compatible with original TSOS
  for (std::vector<TrajectoryMeasurement>::const_iterator mea=meas.begin(); mea!=meas.end(); ++mea) {
    hit_num++;
    
    // Update TSOS with measurement on first considered layer
    TrajectoryStateOnSurface updatedTSOS = updator_->update(mea->forwardPredictedState(), *mea->recHit());

    LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::makeSeedsFromHitDoublets: TSOS for TM " << found << std::endl;
    if (not updatedTSOS.isValid()) continue;    // Skip if updated TSOS is invalid

    edm::OwnVector<TrackingRecHit> seedHits;
      
    // Save hit on first layer
    seedHits.push_back(*mea->recHit()->hit());
    const DetLayer* detLayer = dynamic_cast<const DetLayer*>(&layer);


    // // // Now for this measurement we will loop over additional layers and try to update the TSOS again // // //

    // find layers compatible with updated TSOS
    auto const& compLayers = (*navSchool).nextLayers(*detLayer, *updatedTSOS.freeState(), alongMomentum);

    int addtnl_layers_scanned=0;
    int found_compatible_on_next_layer = 0;
    int det_id = 0;
      
    // Copy updated TSOS - we will update it again with a measurement from the next layer, if we find it
    TrajectoryStateOnSurface updatedTSOS_next(updatedTSOS);

    // loop over layers compatible with updated TSOS
    for (auto compLayer : compLayers) {
      int nmeas=0;

      if (addtnl_layers_scanned>=max_addtnl_layers) break;    // break if we already looped over enough layers
      if (found_compatible_on_next_layer>0) break;    // break if we already found additional hit

      // find dets compatible with updated TSOS
      std::vector< GeometricSearchDet::DetWithState > dets_next;
      TrajectoryStateOnSurface onLayer_next(updatedTSOS);
      onLayer_next.rescaleError(errorSF);//errorSF
      compLayer->compatibleDetsV(onLayer_next, propagatorAlong, *estimator, dets_next);

      //if (!detWithState.size()) continue;
      std::vector<TrajectoryMeasurement> meas_next;
      
      // find measurements on dets_next and save the valid ones
      for (std::vector<GeometricSearchDet::DetWithState>::iterator idet_next=dets_next.begin(); idet_next!=dets_next.end(); ++idet_next) {
        MeasurementDetWithData det = measurementTracker->idToDet(idet_next->first->geographicalId());

        if (det.isNull()) continue;    // skip if det does not exist
        if (!idet_next->second.isValid()) continue;    // skip if TSOS is invalid

        // Find measurements on this det
        std::vector <TrajectoryMeasurement>mymeas_next=det.fastMeasurements(idet_next->second, onLayer_next, propagatorAlong, *estimator);

        for (std::vector<TrajectoryMeasurement>::const_iterator imea_next=mymeas_next.begin(), ed2=mymeas_next.end(); imea_next != ed2; ++imea_next) {
            
          // save valid measurements
          if (imea_next->recHit()->isValid()) meas_next.push_back(*imea_next);

        }    // end loop over mymeas_next
      }    // end loop over dets_next

      // sort valid measurements found on this layer
      std::sort(meas_next.begin(), meas_next.end(), TrajMeasLessEstim());
        
      // loop over valid measurements compatible with updated TSOS (TSOS updated with a hit on the first layer)
      for (std::vector<TrajectoryMeasurement>::const_iterator mea_next=meas_next.begin(); mea_next!=meas_next.end(); ++mea_next) {

        if (nmeas>=max_meas) break;    // skip if we already found enough hits
        
        // try to update TSOS again, with an additional hit
        updatedTSOS_next = updator_->update(mea_next->forwardPredictedState(), *mea_next->recHit());
        
        if (not updatedTSOS_next.isValid()) continue;    // skip if TSOS updated with additional hit is not valid

        // If there was a compatible hit on this layer, we end up here.
        // An additional compatible hit is saved.
        seedHits.push_back(*mea_next->recHit()->hit());
        det_id = mea_next->recHit()->geographicalId().rawId();
        nmeas++;
        found_compatible_on_next_layer++;

      } // end loop over meas_next

    addtnl_layers_scanned++;    

    } // end loop over compLayers (additional layers scanned after the original layer)

    if (found_compatible_on_next_layer==0) continue;
    // only consider the hit if there was a compatible hit on one of the additional scanned layers

    // Create a seed from two saved hits
    PTrajectoryStateOnDet const& pstate = trajectoryStateTransform::persistentState(updatedTSOS_next, det_id);
    TrajectorySeed seed(pstate, std::move(seedHits), oppositeToMomentum);

    LogTrace("TSGForOIFromL2") << "TSGForOIFromL2::makeSeedsFromHitDoublets: Number of seedHits: " << seedHits.size() << std::endl;
    out.push_back(seed);

    found++;
    numSeedsMade++;
    hitDoubletSeedsMade++;

    if (found == numOfHitsToTry_) break;    // break if enough measurements scanned
    if (hitDoubletSeedsMade > maxHitDoubletSeeds_) return;    // abort if enough seeds created

  } // end loop over measurements compatible with original TSOS

  if (found)
      layerCount++;

}


//
// Calculate the dynamic error SF by analysing the L2
//
double TSGForOIFromL2::calculateSFFromL2(const reco::TrackRef track) const {
  double theSF = 1.0;
  // L2 direction vs pT blowup - as was previously done:
  // Split into 4 pT ranges: <pT1_, pT1_<pT2_, pT2_<pT3_, <pT4_: 13,30,70
  // Split into different eta ranges depending in pT
  double abseta = std::abs(track->eta());
  if (track->pt() <= pT1_)
    theSF = SF1_;
  else if (track->pt() > pT1_ && track->pt() <= pT2_) {
    if (abseta <= eta3_)
      theSF = SF3_;
    else if (abseta > eta3_ && abseta <= eta6_)
      theSF = SF2_;
    else if (abseta > eta6_)
      theSF = SF3_;
  } else if (track->pt() > pT2_ && track->pt() <= pT3_) {
    if (abseta <= eta1_)
      theSF = SF6_;
    else if (abseta > eta1_ && abseta <= eta2_)
      theSF = SF4_;
    else if (abseta > eta2_ && abseta <= eta3_)
      theSF = SF6_;
    else if (abseta > eta3_ && abseta <= eta4_)
      theSF = SF1_;
    else if (abseta > eta4_ && abseta <= eta5_)
      theSF = SF1_;
    else if (abseta > eta5_)
      theSF = SF5_;
  } else if (track->pt() > pT3_) {
    if (abseta <= eta3_)
      theSF = SF5_;
    else if (abseta > eta3_ && abseta <= eta4_)
      theSF = SF4_;
    else if (abseta > eta4_ && abseta <= eta5_)
      theSF = SF4_;
    else if (abseta > eta5_)
      theSF = SF5_;
  }

  LogTrace(theCategory_) << "TSGForOIFromL2::calculateSFFromL2: SF has been calculated as: " << theSF;

  return theSF;
}

//
// calculate Chi^2 of two trajectory states
//
double TSGForOIFromL2::match_Chi2(const TrajectoryStateOnSurface& tsos1, const TrajectoryStateOnSurface& tsos2) const {
  if (!tsos1.isValid() || !tsos2.isValid())
    return -1.;

  AlgebraicVector5 v(tsos1.localParameters().vector() - tsos2.localParameters().vector());
  AlgebraicSymMatrix55 m(tsos1.localError().matrix() + tsos2.localError().matrix());

  bool ierr = !m.Invert();

  if (ierr) {
    edm::LogInfo("TSGForOIFromL2") << "Error inverting covariance matrix";
    return -1;
  }

  double est = ROOT::Math::Similarity(v, m);

  return est;
}


std::map<std::string, float> TSGForOIFromL2::getFeatureMap(
    reco::TrackRef l2,
    const TrajectoryStateOnSurface& tsos_IP,
    const TrajectoryStateOnSurface& tsos_MuS
) const {
    std::map<std::string, float> the_map;
    the_map["pt"] = l2->pt();
    the_map["eta"] = l2->eta();
    the_map["phi"] = l2->phi();
    the_map["validHits"] = l2->found();
    if (tsos_IP.isValid()) {
        the_map["tsos_IP_eta"] = tsos_IP.globalPosition().eta();
        the_map["tsos_IP_phi"] = tsos_IP.globalPosition().phi();
        the_map["tsos_IP_pt"] = tsos_IP.globalMomentum().perp();
        the_map["tsos_IP_pt_eta"] = tsos_IP.globalMomentum().eta();
        the_map["tsos_IP_pt_phi"] = tsos_IP.globalMomentum().phi();
        AlgebraicSymMatrix55 matrix_IP = tsos_IP.curvilinearError().matrix();
        the_map["err0_IP"] = sqrt(matrix_IP[0][0]);
        the_map["err1_IP"] = sqrt(matrix_IP[1][1]);
        the_map["err2_IP"] = sqrt(matrix_IP[2][2]);
        the_map["err3_IP"] = sqrt(matrix_IP[3][3]);
        the_map["err4_IP"] = sqrt(matrix_IP[4][4]);
        the_map["tsos_IP_valid"] = 1.0;
    } else {
        the_map["tsos_IP_eta"] = -999;
        the_map["tsos_IP_phi"] = -999;
        the_map["tsos_IP_pt"] = -999;
        the_map["tsos_IP_pt_eta"] = -999;
        the_map["tsos_IP_pt_phi"] = -999;
        the_map["err0_IP"] = -999;
        the_map["err1_IP"] = -999;
        the_map["err2_IP"] = -999;
        the_map["err3_IP"] = -999;
        the_map["err4_IP"] = -999;
        the_map["tsos_IP_valid"] = 0.0;
    }
    if (tsos_MuS.isValid()) {
        the_map["tsos_MuS_eta"] = tsos_MuS.globalPosition().eta();
        the_map["tsos_MuS_phi"] = tsos_MuS.globalPosition().phi();
        the_map["tsos_MuS_pt"] = tsos_MuS.globalMomentum().perp();
        the_map["tsos_MuS_pt_eta"] = tsos_MuS.globalMomentum().eta();
        the_map["tsos_MuS_pt_phi"] = tsos_MuS.globalMomentum().phi();
        AlgebraicSymMatrix55 matrix_MuS = tsos_MuS.curvilinearError().matrix();
        the_map["err0_MuS"] = sqrt(matrix_MuS[0][0]);
        the_map["err1_MuS"] = sqrt(matrix_MuS[1][1]);
        the_map["err2_MuS"] = sqrt(matrix_MuS[2][2]);
        the_map["err3_MuS"] = sqrt(matrix_MuS[3][3]);
        the_map["err4_MuS"] = sqrt(matrix_MuS[4][4]);
        the_map["tsos_MuS_valid"] = 1.0;
    } else {
        the_map["tsos_MuS_eta"] = -999;
        the_map["tsos_MuS_phi"] = -999;
        the_map["tsos_MuS_pt"] = -999;
        the_map["tsos_MuS_pt_eta"] = -999;
        the_map["tsos_MuS_pt_phi"] = -999;
        the_map["err0_MuS"] = -999;
        the_map["err1_MuS"] = -999;
        the_map["err2_MuS"] = -999;
        the_map["err3_MuS"] = -999;
        the_map["err4_MuS"] = -999;
        the_map["tsos_MuS_valid"] = 0.0;
    }
    return the_map;
}


void TSGForOIFromL2::evaluateDnn(
    std::map<std::string, float> feature_map,
    tensorflow::Session* session,
    TH1D * inpOrderHist,
    TH1D * layerNamesHist,
    TH2D * decoderHist,
    int& nHB,
    int& nHLIP,
    int& nHLMuS,
    bool& dnnSuccess
) const {
    int n_features = inpOrderHist->GetXaxis()->GetNbins();
    
    // Prepare tensor for DNN inputs
    tensorflow::Tensor input(tensorflow::DT_FLOAT, { 1, n_features });
    std::string fname;
    for (int i=0; i<n_features; i++){
        fname = inpOrderHist->GetXaxis()->GetBinLabel(i+1);
        if (feature_map.find(fname) == feature_map.end()) {
            std::cout << "Couldn't find " << fname << " in feature_map! Will not evaluate DNN." << std::endl;
            return;
        }
        else {
            input.matrix<float>()(0, i) = float(feature_map.at(fname));
            //std::cout << "Input #" << i << ": " << fname << " = " << feature_map.at(fname) << std::endl;
        }
    }

    // Prepare tensor for DNN outputs
    std::vector<tensorflow::Tensor> outputs;

    // Evaluate DNN and put results in output tensor
    std::string inputLayer = layerNamesHist->GetXaxis()->GetBinLabel(1);
    std::string outputLayer = layerNamesHist->GetXaxis()->GetBinLabel(2);
    //std::cout << inputLayer << " " << outputLayer << std::endl;
    tensorflow::run(session, { { inputLayer, input } }, { outputLayer }, &outputs);
    tensorflow::Tensor out_tensor = outputs[0];
    tensorflow::TTypes<float, 1>::Matrix dnn_outputs = out_tensor.matrix<float>();

    // Find output with largest prediction
    int imax = -1;
    float out_max = 0;
    for (long long int i = 0; i < out_tensor.dim_size(1); i++) {
        float ith_output = dnn_outputs(0, i);
        //std::cout << outputLayer << "#" <<  i << " = " << ith_output << std::endl;
        if (ith_output > out_max){
            imax = i;
            out_max = ith_output;
        }
    }

    // Decode output
    nHB = decoderHist->GetBinContent(1, imax+1);
    nHLIP = decoderHist->GetBinContent(2, imax+1);
    nHLMuS = decoderHist->GetBinContent(3, imax+1);

    // If you want to verify that parameters are interpreted correctly,
    // you can print out parameter names, stored as bin lables
    /*for (int i=0; i<decoderHist->GetXaxis()->GetNbins(); i++){
        std::cout << decoderHist->GetXaxis()->GetBinLabel(i+1) << std::endl;
    }*/
    
    //std::cout << "DNN output #"<< imax << ": " << nHB << " " << nHLIP << " " << nHLMuS << std::endl;
    dnnSuccess = true;
    return;
}


//
//
//
void TSGForOIFromL2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltL2Muons", "UpdatedAtVtx"));
  desc.add<int>("layersToTry", 2);
  desc.add<double>("fixedErrorRescaleFactorForHitless", 2.0);
  desc.add<int>("hitsToTry", 1);
  desc.add<bool>("adjustErrorsDynamicallyForHits", false);
  desc.add<bool>("adjustErrorsDynamicallyForHitless", true);
  desc.add<edm::InputTag>("MeasurementTrackerEvent", edm::InputTag("hltSiStripClusters"));
  desc.add<bool>("UseHitLessSeeds", true);
  desc.add<std::string>("estimator", "hltESPChi2MeasurementEstimator100");
  desc.add<double>("maxEtaForTOB", 1.8);
  desc.add<double>("minEtaForTEC", 0.7);
  desc.addUntracked<bool>("debug", false);
  desc.add<double>("fixedErrorRescaleFactorForHits", 1.0);
  desc.add<unsigned int>("maxSeeds", 20);
  desc.add<unsigned int>("maxHitlessSeeds", 5);  
  desc.add<unsigned int>("maxHitSeeds", 1);
  desc.add<unsigned int>("numL2ValidHitsCutAllEta", 20);
  desc.add<unsigned int>("numL2ValidHitsCutAllEndcap", 30);
  desc.add<double>("pT1", 13.0);
  desc.add<double>("pT2", 30.0);
  desc.add<double>("pT3", 70.0);
  desc.add<double>("eta1", 0.2);
  desc.add<double>("eta2", 0.3);
  desc.add<double>("eta3", 1.0);
  desc.add<double>("eta4", 1.2);
  desc.add<double>("eta5", 1.6);
  desc.add<double>("eta6", 1.4);
  desc.add<double>("eta7", 2.1);
  desc.add<double>("SF1", 3.0);
  desc.add<double>("SF2", 4.0);
  desc.add<double>("SF3", 5.0);
  desc.add<double>("SF4", 7.0);
  desc.add<double>("SF5", 10.0);
  desc.add<double>("SF6", 2.0);
  desc.add<double>("tsosDiff1", 0.2);
  desc.add<double>("tsosDiff2", 0.02);
  desc.add<std::string>("propagatorName", "PropagatorWithMaterialParabolicMf");
  desc.add<bool>("useBothAsInRun2", true);
  desc.add<bool>("dontCreateHitbasedInBarrelAsInRun2", true);
  desc.add<unsigned int>("maxHitlessSeedsIP", 5);
  desc.add<unsigned int>("maxHitlessSeedsMuS", 0);
  desc.add<unsigned int>("maxHitDoubletSeeds", 0);
  desc.add<bool>("getStrategyFromDNN", false);
  desc.add<double>("etaSplitForDnn", 1.0);
  desc.add<std::string>("dnnModelPath_barrel", "");
  desc.add<std::string>("dnnMetadataPath_barrel", "");
  desc.add<std::string>("dnnModelPath_endcap", "");
  desc.add<std::string>("dnnMetadataPath_endcap", "");
  descriptions.add("TSGForOIFromL2", desc);
}

DEFINE_FWK_MODULE(TSGForOIFromL2);
