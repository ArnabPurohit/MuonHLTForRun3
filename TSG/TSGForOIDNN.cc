/**
  \class    TSGForOIDNN
  \brief    Create L3MuonTrajectorySeeds from L2 Muons in an outside-in manner
  \author   Dmitry Kondratyev, Arnab Purohit, Jan-Frederik Schulte (Purdue University, West Lafayette, USA)
 */

#include "RecoMuon/TrackerSeedGenerator/plugins/TSGForOIDNN.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <memory>

TSGForOIDNN::TSGForOIDNN(const edm::ParameterSet& iConfig)
    : src_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      t_estimatorH_(esConsumes<Chi2MeasurementEstimatorBase, TrackingComponentsRecord>(edm::ESInputTag("", iConfig.getParameter<std::string>("estimator")))),
      t_magfieldH_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
      t_propagatorAlongH_(esConsumes<Propagator, TrackingComponentsRecord>(edm::ESInputTag("", iConfig.getParameter<std::string>("propagatorName")))),
      t_propagatorOppositeH_(esConsumes<Propagator, TrackingComponentsRecord>(edm::ESInputTag("", iConfig.getParameter<std::string>("propagatorName")))),
      t_tmpTkGeometryH_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      t_geometryH_(esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord>()),
      t_navSchool_(esConsumes<NavigationSchool, NavigationSchoolRecord>(edm::ESInputTag("", "SimpleNavigationSchool"))),
      t_SHPOpposite_(esConsumes<Propagator, TrackingComponentsRecord>(edm::ESInputTag("", "hltESPSteppingHelixPropagatorOpposite"))),
      maxSeeds_(iConfig.getParameter<uint32_t>("maxSeeds")),
      maxHitSeeds_(iConfig.getParameter<uint32_t>("maxHitSeeds")),
      maxHitlessSeeds_(iConfig.getParameter<uint32_t>("maxHitlessSeeds")),
      numOfLayersToTry_(iConfig.getParameter<int32_t>("layersToTry")),
      numOfHitsToTry_(iConfig.getParameter<int32_t>("hitsToTry")),
      fixedErrorRescalingForHitless_(iConfig.getParameter<double>("fixedErrorRescaleFactorForHitless")),
      minEtaForTEC_(iConfig.getParameter<double>("minEtaForTEC")),
      maxEtaForTOB_(iConfig.getParameter<double>("maxEtaForTOB")),
      updator_(new KFUpdator()),
      measurementTrackerTag_(consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>("MeasurementTrackerEvent"))),
      theCategory_(std::string("Muon|RecoMuon|TSGForOIDNN")),
      maxHitlessSeedsIP_(iConfig.getParameter<uint32_t>("maxHitlessSeedsIP")),
      maxHitlessSeedsMuS_(iConfig.getParameter<uint32_t>("maxHitlessSeedsMuS")),
      maxHitDoubletSeeds_(iConfig.getParameter<uint32_t>("maxHitDoubletSeeds")),
      getStrategyFromDNN_(iConfig.getParameter<bool>("getStrategyFromDNN")),
      dnnMetadataPath_(iConfig.getParameter<std::string>("dnnMetadataPath"))
      {
          if (getStrategyFromDNN_){
              edm::FileInPath dnnMetadataPath(dnnMetadataPath_);
              pt::read_json(dnnMetadataPath.fullPath(), metadata);
              tensorflow::setLogging("2");

	      dnnModelPath_HB_ = metadata.get<std::string>("HB.dnnmodel_name");
	      edm::FileInPath dnnPath_HB(dnnModelPath_HB_);
	      graphDef_HB_ = tensorflow::loadGraphDef(dnnPath_HB.fullPath());
	      tf_session_HB_ = tensorflow::createSession(graphDef_HB_);
	      
	      dnnModelPath_HLIP_ = metadata.get<std::string>("HLIP.dnnmodel_name");
	      edm::FileInPath dnnPath_HLIP(dnnModelPath_HLIP_);
	      graphDef_HLIP_ = tensorflow::loadGraphDef(dnnPath_HLIP.fullPath());
	      tf_session_HLIP_ = tensorflow::createSession(graphDef_HLIP_);
	      
	      dnnModelPath_HLMuS_ = metadata.get<std::string>("HLMuS.dnnmodel_name");
	      edm::FileInPath dnnPath_HLMuS(dnnModelPath_HLMuS_);
	      graphDef_HLMuS_ = tensorflow::loadGraphDef(dnnPath_HLMuS.fullPath());
	      tf_session_HLMuS_ = tensorflow::createSession(graphDef_HLMuS_);
          }
          produces<std::vector<TrajectorySeed> >();
      }

TSGForOIDNN::~TSGForOIDNN() {
    if (getStrategyFromDNN_){
      tensorflow::closeSession(tf_session_HB_);
      tensorflow::closeSession(tf_session_HLIP_);
      tensorflow::closeSession(tf_session_HLMuS_);
      delete graphDef_HB_;
      delete graphDef_HLIP_;
      delete graphDef_HLMuS_;
    }
}

//
// Produce seeds
//
void TSGForOIDNN::produce(edm::StreamID sid, edm::Event& iEvent, edm::EventSetup const& iEventSetup) const {
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
    edm::ESHandle<Chi2MeasurementEstimatorBase> estimatorH = iEventSetup.getHandle(t_estimatorH_);
    edm::ESHandle<MagneticField> magfieldH = iEventSetup.getHandle(t_magfieldH_);
    edm::ESHandle<Propagator> propagatorAlongH = iEventSetup.getHandle(t_propagatorAlongH_);
    edm::ESHandle<Propagator> propagatorOppositeH = iEventSetup.getHandle(t_propagatorOppositeH_);
    edm::ESHandle<TrackerGeometry> tmpTkGeometryH = iEventSetup.getHandle(t_tmpTkGeometryH_);
    edm::ESHandle<GlobalTrackingGeometry> geometryH = iEventSetup.getHandle(t_geometryH_);
    edm::ESHandle<NavigationSchool> navSchool = iEventSetup.getHandle(t_navSchool_);

    edm::Handle<MeasurementTrackerEvent> measurementTrackerH;
    iEvent.getByToken(measurementTrackerTag_, measurementTrackerH);

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
    edm::ESHandle<Propagator> SHPOpposite = iEventSetup.getHandle(t_SHPOpposite_);

    // Loop over the L2's and make seeds for all of them
    LogTrace(theCategory_) << "TSGForOIDNN::produce: Number of L2's: " << l2TrackCol->size();
    for (unsigned int l2TrackColIndex(0); l2TrackColIndex != l2TrackCol->size(); ++l2TrackColIndex) {
        const reco::TrackRef l2(l2TrackCol, l2TrackColIndex);

        // Container of Seeds
        std::vector<TrajectorySeed> out;
        LogTrace("TSGForOIDNN") << "TSGForOIDNN::produce: L2 muon pT, eta, phi --> " << l2->pt() << " , " << l2->eta()
            << " , " << l2->phi() << std::endl;

        FreeTrajectoryState fts = trajectoryStateTransform::initialFreeState(*l2, magfieldH.product());

        dummyPlane->move(fts.position() - dummyPlane->position());
        TrajectoryStateOnSurface tsosAtIP = TrajectoryStateOnSurface(fts, *dummyPlane);
        LogTrace("TSGForOIDNN") << "TSGForOIDNN::produce: Created TSOSatIP: " << tsosAtIP << std::endl;

        // Get the TSOS on the innermost layer of the L2
        TrajectoryStateOnSurface tsosAtMuonSystem =
            trajectoryStateTransform::innerStateOnSurface(*l2, *geometryH, magfieldH.product());
        LogTrace("TSGForOIDNN") << "TSGForOIDNN::produce: Created TSOSatMuonSystem: " << tsosAtMuonSystem
            << std::endl;

        LogTrace("TSGForOIDNN")
            << "TSGForOIDNN::produce: Check the error of the L2 parameter and use hit seeds if big errors" << std::endl;

        StateOnTrackerBound fromInside(propagatorAlong.get());
        TrajectoryStateOnSurface outerTkStateInside = fromInside(fts);

        StateOnTrackerBound fromOutside(&*SHPOpposite);
        TrajectoryStateOnSurface outerTkStateOutside = fromOutside(tsosAtMuonSystem);

        // Check if the two positions (using updated and not-updated TSOS) agree withing certain extent.
        // If both TSOSs agree, use only the one at vertex, as it uses more information. If they do not agree, search for seeds based on both.
        double L2muonEta = l2->eta();
        double absL2muonEta = std::abs(L2muonEta);

        // make non-const copies of parameters
        // (we want to override them if DNN evaluation is enabled)
        unsigned int maxHitSeeds__ = maxHitSeeds_;
        unsigned int maxHitDoubletSeeds__ = maxHitDoubletSeeds_;
        unsigned int maxHitlessSeedsIP__ = maxHitlessSeedsIP_;
        unsigned int maxHitlessSeedsMuS__ = maxHitlessSeedsMuS_; 

	float errorSFHitless = fixedErrorRescalingForHitless_;

        // update strategy parameters by evaluating DNN
        if (getStrategyFromDNN_){
            int nHBd(0), nHLIP(0), nHLMuS(0);
	    float SF = 1.0;

            bool dnnSuccess_ = false;

            // Put variables needed for DNN into an std::map
            std::map<std::string, float> feature_map_ = getFeatureMap(l2, tsosAtIP, outerTkStateOutside);

	    std::tie(nHBd, nHLIP, nHLMuS, SF, dnnSuccess_) =  evaluateDnn(feature_map_, tf_session_HB_, metadata.get_child("HB"),tf_session_HLIP_, metadata.get_child("HLIP"),tf_session_HLMuS_, metadata.get_child("HLMuS") );

            if (!dnnSuccess_) break;
	    LogTrace("TSGForOIDNN") << "For L2 muon with pt:"<<l2->pt()<<", eta:"<<l2->eta()<<", phi:"<<l2->phi()<<" DNN decision: " << nHBd << " " << nHLIP << " " << nHLMuS <<" "<<SF<< std::endl;
            maxHitSeeds__ = 0;
            maxHitDoubletSeeds__ = nHBd;
            maxHitlessSeedsIP__ = nHLIP;
            maxHitlessSeedsMuS__ = nHLMuS;
	    errorSFHitless = SF;
        }

        numSeedsMade = 0;
        hitlessSeedsMadeIP = 0;
        hitlessSeedsMadeMuS = 0;
        hitSeedsMade = 0;
        hitDoubletSeedsMade = 0;

        // BARREL
        if (absL2muonEta < maxEtaForTOB_) {
            layerCount = 0;
            for (auto it = tob.rbegin(); it != tob.rend(); ++it) {
                LogTrace("TSGForOIDNN") << "TSGForOIDNN::produce: looping in TOB layer " << layerCount << std::endl;

                if (hitlessSeedsMadeIP < maxHitlessSeedsIP__ && numSeedsMade < maxSeeds_)
                    makeSeedsWithoutHits(**it,
                                         tsosAtIP,
                                         *(propagatorAlong.get()),
                                         estimatorH,
                                         errorSFHitless,
                                         hitlessSeedsMadeIP,
                                         numSeedsMade,
                                         out);

                if (outerTkStateInside.isValid() && outerTkStateOutside.isValid() &&
                    hitlessSeedsMadeMuS < maxHitlessSeedsMuS__ && numSeedsMade < maxSeeds_)
                    makeSeedsWithoutHits(**it,
                                         outerTkStateOutside,
                                         *(propagatorOpposite.get()),
                                         estimatorH,
                                         errorSFHitless,
                                         hitlessSeedsMadeMuS,
                                         numSeedsMade,
                                         out);

                if (hitSeedsMade < maxHitSeeds__ && numSeedsMade < maxSeeds_)
                    makeSeedsFromHits(**it,
                                      tsosAtIP,
                                      *(propagatorAlong.get()),
                                      estimatorH,
                                      measurementTrackerH,
                                      hitSeedsMade,
                                      numSeedsMade,
                                      layerCount,
                                      out);

                if (hitDoubletSeedsMade < maxHitDoubletSeeds__ && numSeedsMade < maxSeeds_)
                    makeSeedsFromHitDoublets(**it,
                                             tsosAtIP,
                                             *(propagatorAlong.get()),
                                             estimatorH,
                                             measurementTrackerH,
                                             navSchool,
                                             hitDoubletSeedsMade,
                                             numSeedsMade,
                                             layerCount,
                                             out);

            }
            LogTrace("TSGForOIDNN") << "TSGForOIDNN:::produce: NumSeedsMade = " << numSeedsMade
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
                LogTrace("TSGForOIDNN") << "TSGForOIDNN::produce: looping in TEC+ layer " << layerCount << std::endl;

                if (hitlessSeedsMadeIP < maxHitlessSeedsIP__ && numSeedsMade < maxSeeds_)
                    makeSeedsWithoutHits(**it,
                                         tsosAtIP,
                                         *(propagatorAlong.get()),
                                         estimatorH,
                                         errorSFHitless,
                                         hitlessSeedsMadeIP,
                                         numSeedsMade,
                                         out);

                if (outerTkStateInside.isValid() && outerTkStateOutside.isValid() &&
                    hitlessSeedsMadeMuS < maxHitlessSeedsMuS__ && numSeedsMade < maxSeeds_)
                    makeSeedsWithoutHits(**it,
                                         outerTkStateOutside,
                                         *(propagatorOpposite.get()),
                                         estimatorH,
                                         errorSFHitless,
                                         hitlessSeedsMadeMuS,
                                         numSeedsMade,
                                         out);

                if (hitSeedsMade < maxHitSeeds__ && numSeedsMade < maxSeeds_)
                    makeSeedsFromHits(**it,
                                      tsosAtIP,
                                      *(propagatorAlong.get()),
                                      estimatorH,
                                      measurementTrackerH,
                                      hitSeedsMade,
                                      numSeedsMade,
                                      layerCount,
                                      out);

                if (hitDoubletSeedsMade < maxHitDoubletSeeds__ && numSeedsMade < maxSeeds_)
                    makeSeedsFromHitDoublets(**it,
                                             tsosAtIP,
                                             *(propagatorAlong.get()),
                                             estimatorH,
                                             measurementTrackerH,
                                             navSchool,
                                             hitDoubletSeedsMade,
                                             numSeedsMade,
                                             layerCount,
                                             out);

            }
            LogTrace("TSGForOIDNN") << "TSGForOIDNN:::produce: NumSeedsMade = " << numSeedsMade
                << " , layerCount = " << layerCount << std::endl;
        }

        // ENDCAP-
        if (L2muonEta < -minEtaForTEC_) {
            layerCount = 0;
            for (auto it = tecNegative.rbegin(); it != tecNegative.rend(); ++it) {
                LogTrace("TSGForOIDNN") << "TSGForOIDNN::produce: looping in TEC- layer " << layerCount << std::endl;

                if (hitlessSeedsMadeIP < maxHitlessSeedsIP__ && numSeedsMade < maxSeeds_)
                    makeSeedsWithoutHits(**it,
                                         tsosAtIP,
                                         *(propagatorAlong.get()),
                                         estimatorH,
                                         errorSFHitless,
                                         hitlessSeedsMadeIP,
                                         numSeedsMade,
                                         out);

                if (outerTkStateInside.isValid() && outerTkStateOutside.isValid() &&
                    hitlessSeedsMadeMuS < maxHitlessSeedsMuS__ && numSeedsMade < maxSeeds_)
                    makeSeedsWithoutHits(**it,
                                         outerTkStateOutside,
                                         *(propagatorOpposite.get()),
                                         estimatorH,
                                         errorSFHitless,
                                         hitlessSeedsMadeMuS,
                                         numSeedsMade,
                                         out);

                if (hitSeedsMade < maxHitSeeds__ && numSeedsMade < maxSeeds_)
                    makeSeedsFromHits(**it,
                                      tsosAtIP,
                                      *(propagatorAlong.get()),
                                      estimatorH,
                                      measurementTrackerH,
                                      hitSeedsMade,
                                      numSeedsMade,
                                      layerCount,
                                      out);

                if (hitDoubletSeedsMade < maxHitDoubletSeeds__ && numSeedsMade < maxSeeds_)
                    makeSeedsFromHitDoublets(**it,
                                             tsosAtIP,
                                             *(propagatorAlong.get()),
                                             estimatorH,
                                             measurementTrackerH,
                                             navSchool,
                                             hitDoubletSeedsMade,
                                             numSeedsMade,
                                             layerCount,
                                             out);

            }
            LogTrace("TSGForOIDNN") << "TSGForOIDNN:::produce: NumSeedsMade = " << numSeedsMade
                << " , layerCount = " << layerCount << std::endl;
        }

        for (std::vector<TrajectorySeed>::iterator it = out.begin(); it != out.end(); ++it) {
            result->push_back(*it);
        }

    }  // L2Collection

    edm::LogInfo(theCategory_) << "TSGForOIDNN::produce: number of seeds made: " << result->size();

    iEvent.put(std::move(result));
}

//
// Create seeds without hits on a given layer (TOB or TEC)
//
void TSGForOIDNN::makeSeedsWithoutHits(const GeometricSearchDet& layer,
                                          const TrajectoryStateOnSurface& tsos,
                                          const Propagator& propagatorAlong,
                                          edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
                                          double errorSF,
                                          unsigned int& hitlessSeedsMade,
                                          unsigned int& numSeedsMade,
                                          std::vector<TrajectorySeed>& out) const {
    // create hitless seeds
    LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsWithoutHits: Start hitless" << std::endl;
    std::vector<GeometricSearchDet::DetWithState> dets;
    layer.compatibleDetsV(tsos, propagatorAlong, *estimator, dets);
    if (!dets.empty()) {
        auto const& detOnLayer = dets.front().first;
        auto const& tsosOnLayer = dets.front().second;
        LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsWithoutHits: tsosOnLayer " << tsosOnLayer << std::endl;
        if (!tsosOnLayer.isValid()) {
            edm::LogInfo(theCategory_) << "ERROR!: Hitless TSOS is not valid!";
        } else {
            dets.front().second.rescaleError(errorSF);
            PTrajectoryStateOnDet const& ptsod =
                trajectoryStateTransform::persistentState(tsosOnLayer, detOnLayer->geographicalId().rawId());
            TrajectorySeed::RecHitContainer rHC;
            out.push_back(TrajectorySeed(ptsod, rHC, oppositeToMomentum));
            LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsWithoutHits: TSOS (Hitless) done " << std::endl;
            hitlessSeedsMade++;
            numSeedsMade++;
        }
    }
}

//
// Find hits on a given layer (TOB or TEC) and create seeds from updated TSOS with hit
//
void TSGForOIDNN::makeSeedsFromHits(const GeometricSearchDet& layer,
                                       const TrajectoryStateOnSurface& tsos,
                                       const Propagator& propagatorAlong,
                                       edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
                                       edm::Handle<MeasurementTrackerEvent>& measurementTracker,
                                       unsigned int& hitSeedsMade,
                                       unsigned int& numSeedsMade,
                                       unsigned int& layerCount,
                                       std::vector<TrajectorySeed>& out) const {
    if (layerCount > numOfLayersToTry_)
        return;

    TrajectoryStateOnSurface onLayer(tsos);

    std::vector<GeometricSearchDet::DetWithState> dets;
    layer.compatibleDetsV(onLayer, propagatorAlong, *estimator, dets);

    // Find Measurements on each DetWithState
    LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHits: Find measurements on each detWithState  "
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
    LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHits: Update TSOS using TMs after sorting, then create "
        "Trajectory Seed, number of TM = "
        << meas.size() << std::endl;
    std::sort(meas.begin(), meas.end(), TrajMeasLessEstim());

    unsigned int found = 0;
    for (std::vector<TrajectoryMeasurement>::const_iterator it = meas.begin(); it != meas.end(); ++it) {
        TrajectoryStateOnSurface updatedTSOS = updator_->update(it->forwardPredictedState(), *it->recHit());
        LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHits: TSOS for TM " << found << std::endl;
        if (not updatedTSOS.isValid())
            continue;

        edm::OwnVector<TrackingRecHit> seedHits;
        seedHits.push_back(*it->recHit()->hit());
        PTrajectoryStateOnDet const& pstate =
            trajectoryStateTransform::persistentState(updatedTSOS, it->recHit()->geographicalId().rawId());
        LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHits: Number of seedHits: " << seedHits.size()
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



void TSGForOIDNN::makeSeedsFromHitDoublets(const GeometricSearchDet& layer,
                                              const TrajectoryStateOnSurface& tsos,
                                              const Propagator& propagatorAlong,
                                              edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
                                              edm::Handle<MeasurementTrackerEvent>& measurementTracker,
                                              edm::ESHandle<NavigationSchool> navSchool,
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

    TrajectoryStateOnSurface onLayer(tsos);

    // Find dets compatible with original TSOS
    std::vector< GeometricSearchDet::DetWithState > dets;
    layer.compatibleDetsV(onLayer, propagatorAlong, *estimator, dets);

    LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHitDoublets: Find measurements on each detWithState  " << dets.size() << std::endl;
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

    LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHitDoublets: Update TSOS using TMs after sorting, then create Trajectory Seed, number of TM = " << meas.size() << std::endl;

    // sort valid measurements found on the first layer
    std::sort(meas.begin(), meas.end(), TrajMeasLessEstim());

    unsigned int found = 0;
    int hit_num = 0;

    // Loop over all valid measurements compatible with original TSOS
    for (std::vector<TrajectoryMeasurement>::const_iterator mea=meas.begin(); mea!=meas.end(); ++mea) {
        hit_num++;

        // Update TSOS with measurement on first considered layer
        TrajectoryStateOnSurface updatedTSOS = updator_->update(mea->forwardPredictedState(), *mea->recHit());

        LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHitDoublets: TSOS for TM " << found << std::endl;
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

        LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHitDoublets: Number of seedHits: " << seedHits.size() << std::endl;
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


std::map<std::string, float> TSGForOIDNN::getFeatureMap(
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


std::tuple<int, int, int, float,  bool> TSGForOIDNN::evaluateDnn(
    std::map<std::string, float> feature_map,
    tensorflow::Session* session_HB,
    const pt::ptree& metadata_HB,
    tensorflow::Session* session_HLIP,
    const pt::ptree& metadata_HLIP,
    tensorflow::Session* session_HLMuS,
    const pt::ptree& metadata_HLMuS
) const {
    int nHBd, nHLIP,nHLMuS, n_features = 0;
    float SF = 1.0;
    bool dnnSuccess = false;
    n_features = metadata_HB.get<int>("n_features", 0);

    // Prepare tensor for DNN inputs
    tensorflow::Tensor input(tensorflow::DT_FLOAT, { 1, n_features });
    std::string fname;
    int i_feature = 0;
    for (const pt::ptree::value_type &feature : metadata_HB.get_child("feature_names")){
        fname =  feature.second.data();
        if (feature_map.find(fname) == feature_map.end()) {
	    // don't evaluate DNN if any input feature is missing
            LogTrace("TSGForOIDNN") << "Couldn't find " << fname << " in feature_map! Will not evaluate DNN." << std::endl;
            return std::make_tuple(nHBd, nHLIP, nHLMuS, SF, dnnSuccess);
        }
        else {
            input.matrix<float>()(0, i_feature) = float(feature_map.at(fname));
            i_feature++;
        }
    }

    // Prepare tensor for DNN outputs
    std::vector<tensorflow::Tensor> outputs_HB;

    // Evaluate DNN and put results in output tensor
    std::string inputLayer_HB = metadata_HB.get<std::string>("input_layer");
    std::string outputLayer_HB = metadata_HB.get<std::string>("output_layer");
    tensorflow::run(session_HB, { { inputLayer_HB, input } }, { outputLayer_HB }, &outputs_HB);
    tensorflow::Tensor out_tensor_HB = outputs_HB[0];
    tensorflow::TTypes<float, 1>::Matrix dnn_outputs_HB = out_tensor_HB.matrix<float>();


    std::vector<tensorflow::Tensor> outputs_HLIP;

    // Evaluate DNN and put results in output tensor
    std::string inputLayer_HLIP = metadata_HLIP.get<std::string>("input_layer");
    std::string outputLayer_HLIP = metadata_HLIP.get<std::string>("output_layer");
    tensorflow::run(session_HLIP, { { inputLayer_HLIP, input } }, { outputLayer_HLIP }, &outputs_HLIP);
    tensorflow::Tensor out_tensor_HLIP = outputs_HLIP[0];
    tensorflow::TTypes<float, 1>::Matrix dnn_outputs_HLIP = out_tensor_HLIP.matrix<float>();

    std::vector<tensorflow::Tensor> outputs_HLMuS;

    // Evaluate DNN and put results in output tensor
    std::string inputLayer_HLMuS = metadata_HLMuS.get<std::string>("input_layer");
    std::string outputLayer_HLMuS = metadata_HLMuS.get<std::string>("output_layer");
    tensorflow::run(session_HLMuS, { { inputLayer_HLMuS, input } }, { outputLayer_HLMuS }, &outputs_HLMuS);
    tensorflow::Tensor out_tensor_HLMuS = outputs_HLMuS[0];
    tensorflow::TTypes<float, 1>::Matrix dnn_outputs_HLMuS = out_tensor_HLMuS.matrix<float>();

    // Decode output
    nHBd = round(dnn_outputs_HB(0, 0)); 
    nHLIP = round(dnn_outputs_HLIP(0, 0));
    SF = round(dnn_outputs_HLIP(0, 1));
    nHLMuS = round(dnn_outputs_HLMuS(0, 0));
    if(nHBd<0) nHBd = 0;
    if(nHLIP<0) nHLIP = 0;
    if(nHLMuS<0) nHLMuS = 0;
    if(SF<0) SF = 0;
    if(nHBd==0 && nHLIP==0 && nHLMuS==0){
      nHBd=1;
      nHLIP=5;
      LogTrace("TSGForOIDNN") <<"DNN prediction for all Hit types are 0. Will choose nHBd=1, nHLIP=5 and nHLMuS=0"<<std::endl;
    }
    if(SF==0){
      SF=1;
      LogTrace("TSGForOIDNN") <<"DNN prediction for SF is 0. Using 1 as default."<<std::endl;
    }
    
    dnnSuccess = true;
    return std::make_tuple(nHBd, nHLIP, nHLMuS, SF, dnnSuccess);
}


//
//
//
void TSGForOIDNN::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltL2Muons", "UpdatedAtVtx"));
  desc.add<int>("layersToTry", 2);
  desc.add<double>("fixedErrorRescaleFactorForHitless", 2.0);
  desc.add<int>("hitsToTry", 1);
  desc.add<edm::InputTag>("MeasurementTrackerEvent", edm::InputTag("hltSiStripClusters"));
  desc.add<std::string>("estimator", "hltESPChi2MeasurementEstimator100");
  desc.add<double>("maxEtaForTOB", 1.8);
  desc.add<double>("minEtaForTEC", 0.7);
  desc.addUntracked<bool>("debug", false);
  desc.add<unsigned int>("maxSeeds", 20);
  desc.add<unsigned int>("maxHitlessSeeds", 5);  
  desc.add<unsigned int>("maxHitSeeds", 1);
  desc.add<std::string>("propagatorName", "PropagatorWithMaterialParabolicMf");
  desc.add<unsigned int>("maxHitlessSeedsIP", 5);
  desc.add<unsigned int>("maxHitlessSeedsMuS", 0);
  desc.add<unsigned int>("maxHitDoubletSeeds", 0);
  desc.add<bool>("getStrategyFromDNN", false);
  desc.add<std::string>("dnnMetadataPath", "");
  descriptions.add("TSGForOIDNN", desc);
}

DEFINE_FWK_MODULE(TSGForOIDNN);
