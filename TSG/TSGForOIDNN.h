#ifndef RecoMuon_TrackerSeedGenerator_TSGForOIDNN_H
#define RecoMuon_TrackerSeedGenerator_TSGForOIDNN_H

/**
 \class    TSGForOIDNN
 \brief    Create L3MuonTrajectorySeeds from L2 Muons updated at vertex in an outside-in manner
 \author   Dmitry Kondratyev, Arnab Purohit, Jan-Frederik Schulte (Purdue University, West Lafayette)
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
namespace pt = boost::property_tree;


class TSGForOIDNN : public edm::global::EDProducer<> {
    public:
    explicit TSGForOIDNN(const edm::ParameterSet& iConfig);
    ~TSGForOIDNN() override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    void produce(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

    private:
    /// Labels for input collections
    const edm::EDGetTokenT<reco::TrackCollection> src_;

    /// Tokens for ESHandle
    const edm::ESGetToken<Chi2MeasurementEstimatorBase, TrackingComponentsRecord> t_estimatorH_;
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> t_magfieldH_;
    const edm::ESGetToken<Propagator, TrackingComponentsRecord> t_propagatorAlongH_;
    const edm::ESGetToken<Propagator, TrackingComponentsRecord> t_propagatorOppositeH_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> t_tmpTkGeometryH_;
    const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> t_geometryH_;
    const edm::ESGetToken<NavigationSchool, NavigationSchoolRecord> t_navSchool_;
    const edm::ESGetToken<Propagator, TrackingComponentsRecord> t_SHPOpposite_;

    /// Maximum number of seeds for each L2
    const unsigned int maxSeeds_;

    /// Maximum number of hitbased seeds for each L2
    const unsigned int maxHitSeeds_;

    /// Maximum number of hitless seeds for each L2
    const unsigned int maxHitlessSeeds_;

    /// How many layers to try
    const unsigned int numOfLayersToTry_;

    /// How many hits to try per layer
    const unsigned int numOfHitsToTry_;

    /// Rescale L2 parameter uncertainties (fixed error vs pT, eta)
    const double fixedErrorRescalingForHitless_;

    /// Estimator used to find dets and TrajectoryMeasurements
    const std::string estimatorName_;

    /// Minimum eta value to activate searching in the TEC
    const double minEtaForTEC_;

    /// Maximum eta value to activate searching in the TOB
    const double maxEtaForTOB_;

    /// KFUpdator defined in constructor
    const std::unique_ptr<TrajectoryStateUpdator> updator_;

    const edm::EDGetTokenT<MeasurementTrackerEvent> measurementTrackerTag_;

    /// Counters and flags for the implementation
    const std::string propagatorName_;
    const std::string theCategory_;

    /// IP refers to TSOS at interaction point, 
    /// MuS refers to TSOS at muon system
    const unsigned int maxHitlessSeedsIP_;
    const unsigned int maxHitlessSeedsMuS_;
    const unsigned int maxHitDoubletSeeds_;

    /// Get number of seeds to use from DNN output instead of "max..Seeds" parameters
    const bool getStrategyFromDNN_;
    std::string dnnModelPath_HB_;
    std::string dnnModelPath_HLIP_;
    std::string dnnModelPath_HLMuS_;

    pt::ptree metadata;
    tensorflow::GraphDef* graphDef_HB_;
    tensorflow::Session* tf_session_HB_;
    
    tensorflow::GraphDef* graphDef_HLIP_;
    tensorflow::Session* tf_session_HLIP_;
    
    tensorflow::GraphDef* graphDef_HLMuS_;
    tensorflow::Session* tf_session_HLMuS_;

    const std::string dnnMetadataPath_;

    /// Create seeds without hits on a given layer (TOB or TEC)
    void makeSeedsWithoutHits(const GeometricSearchDet& layer,
                              const TrajectoryStateOnSurface& tsos,
                              const Propagator& propagatorAlong,
                              edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
                              double errorSF,
                              unsigned int& hitlessSeedsMade,
                              unsigned int& numSeedsMade,
                              std::vector<TrajectorySeed>& out) const;

    /// Find hits on a given layer (TOB or TEC) and create seeds from updated TSOS with hit
    void makeSeedsFromHits(const GeometricSearchDet& layer,
                           const TrajectoryStateOnSurface& tsos,
                           const Propagator& propagatorAlong,
                           edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
                           edm::Handle<MeasurementTrackerEvent>& measurementTracker,
                           unsigned int& hitSeedsMade,
                           unsigned int& numSeedsMade,
                           unsigned int& layerCount,
                           std::vector<TrajectorySeed>& out) const;

    /// Similar to makeSeedsFromHits, but seed is created only if there are compatible hits on two adjacent layers
    void makeSeedsFromHitDoublets(const GeometricSearchDet& layer,
                                  const TrajectoryStateOnSurface& tsos,
                                  const Propagator& propagatorAlong,
                                  edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
                                  edm::Handle<MeasurementTrackerEvent>& measurementTracker,
                                  edm::ESHandle<NavigationSchool> navSchool,
                                  unsigned int& hitDoubletSeedsMade,
                                  unsigned int& numSeedsMade,
                                  unsigned int& layerCount,
                                  std::vector<TrajectorySeed>& out) const;


    /// Dictionary of inputs for DNN
    std::map<std::string, float> getFeatureMap(
        reco::TrackRef l2,
        const TrajectoryStateOnSurface& tsos_IP,
        const TrajectoryStateOnSurface& tsos_MuS
    ) const;

    /// Evaluate DNN
    std::tuple<int, int, int, float, bool>  evaluateDnn(
        std::map<std::string, float> feature_map,
	tensorflow::Session* session_HB,
	const pt::ptree& metadata_HB,
	tensorflow::Session* session_HLIP,
	const pt::ptree& metadata_HLIP,
	tensorflow::Session* session_HLMuS,
	const pt::ptree& metadata_HLMuS
    ) const;

};

#endif
