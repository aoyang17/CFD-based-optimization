'''
---------------------------------------------------------
 
This script is to set up the modal parameterization for ASO. 

This script is updated to compatible with latest version of pyGeo
---------------------------------------------------------
'''

# ======================================================================
#         Imports
# ======================================================================
import copy
from collections import OrderedDict
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from mpi4py import MPI
from pyspline import Curve
from pyspline.utils import openTecplot, closeTecplot, writeTecplot1D, writeTecplot3D
from pygeo import pyNetwork, pyBlock, geo_utils
import os
import warnings
from baseclasses.utils import Error
# from .designVars import geoDVGlobal, geoDVLocal, geoDVSpanwiseLocal, geoDVSectionLocal, geoDVComposite
from pygeo.parameterization.BaseDVGeo import BaseDVGeometry

class DVGeometry_MODE(BaseDVGeometry):
    r"""
    A class for manipulating geometry.
    The purpose of the DVGeometry class is to provide a mapping from
    user-supplied design variables to an arbitrary set of discrete,
    three-dimensional coordinates. These three-dimensional coordinates
    can in general represent anything, but will typically be the
    surface of an aerodynamic mesh, the nodes of a FE mesh or the
    nodes of another geometric construct.
    In a very general sense, DVGeometry performs two primary
    functions:
    1. Given a new set of design variables, update the
       three-dimensional coordinates: :math:`X_{DV}\rightarrow
       X_{pt}` where :math:`X_{pt}` are the coordinates and :math:`X_{DV}`
       are the user variables.
    2. Determine the derivative of the coordinates with respect to the
       design variables. That is the derivative :math:`\frac{dX_{pt}}{dX_{DV}}`
    DVGeometry uses the *Free-Form Deformation* approach for geometry
    manipulation. The basic idea is the coordinates are *embedded* in
    a clear-flexible jelly-like block. Then by stretching moving and
    'poking' the volume, the coordinates that are embedded inside move
    along with overall deformation of the volume.
    Parameters
    ----------
    fileName : str
       filename of FFD file. This must be a ascii formatted plot3D file
       in fortran ordering.
    isComplex : bool
        Make the entire object complex. This should **only** be used when
        debugging the entire tool-chain with the complex step method.
    child : bool
        Flag to indicate that this object is a child of parent DVGeo object
    faceFreeze : dict
        A dictionary of lists of strings specifying which faces should be
        'frozen'. Each dictionary represents one block in the FFD.
        For example if faceFreeze =['0':['iLow'],'1':[]], then the
        plane of control points corresponding to i=0, and i=1, in block '0'
        will not be able to move in DVGeometry.
    name : str
        This is prepended to every DV name for ensuring design variables names are
        unique to pyOptSparse. Only useful when using multiple DVGeos with
        :meth:`.addTriangulatedSurfaceConstraint()`
    kmax : int
        Maximum order of the splines used for the underlying formulation.
        Default is a 4th order spline in each direction if the dimensions
        allow.
    Examples
    --------
    The general sequence of operations for using DVGeometry is as follows::
      >>> from pygeo import DVGeometry
      >>> DVGeo = DVGeometry('FFD_file.fmt')
      >>> # Embed a set of coordinates Xpt into the object
      >>> DVGeo.addPointSet(Xpt, 'myPoints')
      >>> # Associate a 'reference axis' for large-scale manipulation
      >>> DVGeo.addRefAxis('wing_axis', axis_curve)
      >>> # Define a global design variable function:
      >>> def twist(val, geo):
      >>>    geo.rot_z['wing_axis'].coef[:] = val[:]
      >>> # Now add this as a global variable:
      >>> DVGeo.addGlobalDV('wing_twist', 0.0, twist, lower=-10, upper=10)
      >>> # Now add local (shape) variables
      >>> DVGeo.addLocalDV('shape', lower=-0.5, upper=0.5, axis='y')
    """

    def __init__(self, fileName, *args, isComplex=False, child=False, faceFreeze=None, name=None, kmax=4, **kwargs):
        super().__init__(fileName=fileName)

        self.DV_listGlobal  = OrderedDict() # Global Design Variable List
        self.DV_listLocal = OrderedDict() # Local Design Variable List

        # Flags to determine if this DVGeometry is a parent or child
        self.isChild = child
        self.children = []
        self.iChild = None
        self.points = OrderedDict()
        self.updated = {}
        self.masks = OrderedDict()
        self.finalized = False
        self.complex = complex
        if self.complex:
            self.dtype = 'D'
        else:
            self.dtype = 'd'

        self.name = name
        self.useComposite = False
        # Load the FFD file in FFD mode. Also note that args and
        # kwargs are passed through in case additional pyBlock options
        # need to be set. 
        self.FFD = pyBlock('plot3d', fileName=fileName, FFD=True,
                           *args, **kwargs)
        self.origFFDCoef = self.FFD.coef.copy()

        # Jacobians:
        self.ptSetNames = []
        self.JT = {}
        self.nPts = {}
     
        # Derivatives of Xref and Coef provided by the parent to the
        # children
        self.dXrefdXdvg = None
        self.dCoefdXdvg = None

        self.dXrefdXdvl = None
        self.dCoefdXdvl = None
        
        # derivative counters for offsets
        self.nDV_T = None
        self.nDVG_T = None
        self.nDVL_T = None
        self.nDVG_count = 0
        self.nDVL_count = 0

        # The set of user supplied axis. 
        self.axis = OrderedDict()

        self.coefRotM = None
        self.coord_xfer = {}
        
    def addRefAxis(
        self,
        name,
        curve=None,
        xFraction=None,
        yFraction=None,
        zFraction=None,
        volumes=None,
        rotType=5,
        axis="x",
        alignIndex=None,
        rotAxisVar=None,
        rot0ang=None,
        rot0axis=[1, 0, 0],
        includeVols=[],
        ignoreInd=[],
        raySize=1.5,
    ):
        """
        This function is used to add a 'reference' axis to the
        DVGeometry object.  Adding a reference axis is only required
        when 'global' design variables are to be used, i.e. variables
        like span, sweep, chord etc --- variables that affect many FFD
        control points.
        There are two different ways that a reference can be
        specified:
        #. The first is explicitly a pySpline curve object using the
           keyword argument curve=<curve>.
        #. The second is to specify the xFraction variable. There are a
           few caveats with the use of this method. First, DVGeometry
           will try to determine automatically the orientation of the FFD
           volume. Then, a reference axis will consist of the same number of
           control points as the number of span-wise sections in the FFD volume
           and will be oriented in the streamwise (x-direction) according to the
           xPercent keyword argument.
        Parameters
        ----------
        name : str
            Name of the reference axis. This name is used in the
            user-supplied design variable functions to determine what
            axis operations occur on.
        curve : pySpline curve object
            Supply exactly the desired reference axis
        xFraction : float
            Specify the parametric stream-wise (axis: 0) location of the reference axis node relative to
            front and rear control points location. Constant for every spanwise section.
        yFraction : float
            Specify the parametric location of the reference axis node along axis: 1 relative to
            top and bottom control points location. Constant for every spanwise section.
            .. note::
                if this is the spanwise axis of the FFD box, the refAxis node will remain in-plane
                and the option will not have any effect.
        zFraction : float
            Specify the parametric location of the reference axis node along axis: 2 relative to
            top and bottom control points location. Constant for every spanwise section.
            .. note::
                if this is the spanwise axis of the FFD box, the refAxis node will remain in-plane
                and the option will not have any effect.
        volumes : list or array or integers
            List of the volume indices, in 0-based ordering that this
            reference axis should manipulate. If xFraction is
            specified, the volumes argument must contain at most 1
            volume. If the volumes is not given, then all volumes are
            taken.
        rotType : int
            Integer in range 0->6 (inclusive) to determine the order
            that the rotations are made.
            0. Intrinsic rotation, rot_theta is rotation about axis
            1. x-y-z
            2. x-z-y
            3. y-z-x
            4. y-x-z
            5. z-x-y  Default (x-streamwise y-up z-out wing)
            6. z-y-x
            7. z-x-y + rot_theta
            8. z-x-y + rotation about section axis (to allow for winglet rotation)
        axis: str
            Axis along which to project points/control points onto the
            ref axis. Default is `x` which will project rays.
        alignIndex: str
            FFD axis along which the reference axis will lie. Can be `i`, `j`,
            or `k`. Only necessary when using xFraction.
        rotAxisVar: str
            If rotType == 8, then you must specify the name of the section local
            variable which should be used to compute the orientation of the theta
            rotation.
        rot0ang: float
            If rotType == 0, defines the offset angle of the (child) FFD with respect
            to the main system of reference. This is necessary to use the scaling functions
            `scale_x`, `scale_y`, and `scale_z` with rotType == 0. The axis of rotation is
            defined by `rot0axis`.
        rot0axis: list
            If rotType == 0, defines the rotation axis for the rotation offset of the
            FFD grid given by `rot0ang`. The variable has to be a list of 3 floats
            defining the [x,y,z] components of the axis direction.
            This is necessary to use the scaling functions `scale_x`, `scale_y`,
            and `scale_z` with rotType == 0.
        includeVols : list
            List of additional volumes to add to reference axis after the
            automatic generation of the ref axis based on the volumes list using
            xFraction.
        ignoreInd : list
            List of indices that should be ignored from the volumes that were
            added to this reference axis. This can be handy if you have a single
            volume but you want to link different sets of indices to different
            reference axes.
        raySize : float
            Used in projection to find attachment point on reference axis.
            See full description in pyNetwork.projectRays function doc string.
            In most cases the default value is sufficient. In the case of highly
            swept wings its sometimes necessary to increase this value.
        Notes
        -----
        One of curve or xFraction must be specified.
        Examples
        --------
        >>> # Simple wing with single volume FFD, reference axis at 1/4 chord:
        >>> DVGeo.addRefAxis('wing', xFraction=0.25)
        >>> # Multiblock FFD, wing is volume 6.
        >>> DVGeo.addRefAxis('wing', xFraction=0.25, volumes=[6])
        >>> # Multiblock FFD, multiple volumes attached refAxis
        >>> DVGeo.addRefAxis('wing', myCurve, volumes=[2,3,4])
        Returns
        -------
        nAxis : int
            The number of control points on the reference axis.
        """

        # We don't do any of the final processing here; we simply
        # record the information the user has supplied into a
        # dictionary structure.
        if axis is None:
            pass
        elif axis.lower() == "x":
            axis = np.array([1, 0, 0], "d")
        elif axis.lower() == "y":
            axis = np.array([0, 1, 0], "d")
        elif axis.lower() == "z":
            axis = np.array([0, 0, 1], "d")

        if curve is not None:
            # Explicit curve has been supplied:
            if self.FFD.symmPlane is None:
                if volumes is None:
                    volumes = np.arange(self.FFD.nVol)
                self.axis[name] = {
                    "curve": curve,
                    "volumes": volumes,
                    "rotType": rotType,
                    "axis": axis,
                    "rot0ang": rot0ang,
                    "rot0axis": rot0axis,
                }

            else:
                # get the direction of the symmetry plane
                if self.FFD.symmPlane.lower() == "x":
                    index = 0
                elif self.FFD.symmPlane.lower() == "y":
                    index = 1
                elif self.FFD.symmPlane.lower() == "z":
                    index = 2

                # mirror the axis and attach the mirrored vols
                if volumes is None:
                    volumes = np.arange(self.FFD.nVol / 2)

                volumesSymm = []
                for volume in volumes:
                    volumesSymm.append(volume + self.FFD.nVol / 2)

                # We want to create a curve that is symmetric of the current one
                symm_curve_X = curve.X.copy()

                # flip the coefs
                symm_curve_X[:, index] = -symm_curve_X[:, index]
                curveSymm = Curve(k=curve.k, X=symm_curve_X)

                self.axis[name] = {
                    "curve": curve,
                    "volumes": volumes,
                    "rotType": rotType,
                    "axis": axis,
                    "rot0ang": rot0ang,
                    "rot0axis": rot0axis,
                }
                self.axis[name + "Symm"] = {
                    "curve": curveSymm,
                    "volumes": volumesSymm,
                    "rotType": rotType,
                    "axis": axis,
                    "rot0ang": rot0ang,
                    "rot0axis": rot0axis,
                }
            nAxis = len(curve.coef)
        elif xFraction or yFraction or zFraction:
            # Some assumptions
            #   - FFD should be a close approximation of geometry surface so that
            #       xFraction roughly corresponds to airfoil LE, TE, or 1/4 chord
            #   - User provides 'i', 'j' or 'k' to specify which block direction
            #       the reference axis should project
            #   - if no volumes are listed, it is assumed that all volumes are
            #       included
            #   - 'x' is streamwise direction

            # Default to "mean" ref axis location along non-user specified direction

            # This is the block direction along which the reference axis will lie
            # alignIndex = 'k'
            if alignIndex is None:
                raise Error("Must specify alignIndex to use xFraction.")

            # Get index direction along which refaxis will be aligned
            if alignIndex.lower() == "i":
                alignIndex = 0
                faceCol = 2
            elif alignIndex.lower() == "j":
                alignIndex = 1
                faceCol = 4
            elif alignIndex.lower() == "k":
                alignIndex = 2
                faceCol = 0

            if volumes is None:
                volumes = range(self.FFD.nVol)

            # Reorder the volumes in sequential order and check if orientation is correct
            v = list(volumes)
            nVol = len(v)
            volOrd = [v.pop(0)]
            faceLink = self.FFD.topo.faceLink
            for _iter in range(nVol):
                for vInd, i in enumerate(v):
                    for pInd, j in enumerate(volOrd):
                        if faceLink[i, faceCol] == faceLink[j, faceCol + 1]:
                            volOrd.insert(pInd + 1, v.pop(vInd))
                            break
                        elif faceLink[i, faceCol + 1] == faceLink[j, faceCol]:
                            volOrd.insert(pInd, v.pop(vInd))
                            break

            if len(volOrd) < nVol:
                raise Error(
                    "The volumes are not ordered with matching faces" " in the direction of the reference axis."
                )

            # Count total number of sections and check if volumes are aligned
            # face to face along refaxis direction
            # Local indices size is (N_x,N_y,N_z)
            lIndex = self.FFD.topo.lIndex
            nSections = []
            for i in range(len(volOrd)):
                if i == 0:
                    nSections.append(lIndex[volOrd[i]].shape[alignIndex])
                else:
                    nSections.append(lIndex[volOrd[i]].shape[alignIndex] - 1)

            refaxisNodes = np.zeros((sum(nSections), 3))

            # Loop through sections and compute node location
            place = 0
            for j, vol in enumerate(volOrd):
                # sectionArr: indices of FFD points grouped by section - i.e. the first tensor index now == nSections
                sectionArr = np.rollaxis(lIndex[vol], alignIndex, 0)
                skip = 0
                if j > 0:
                    skip = 1
                for i in range(nSections[j]):
                    # getting all the section control points coordinates
                    pts_tens = self.FFD.coef[sectionArr[i + skip, :, :], :]  # shape=(xAxisNodes,yAxisnodes,3)

                    # reshaping into vector to allow rotation (if needed) - leveraging on pts_tens.shape[2]=3 (FFD cp coordinates)
                    pts_vec = np.copy(pts_tens.reshape(-1, 3))  # new shape=(xAxisNodes*yAxisnodes,3)

                    if rot0ang:
                        # rotating the FFD to be aligned with main axes
                        for ct_ in range(np.shape(pts_vec)[0]):
                            # here we loop over the pts_vec, rotate them and insert them inplace in pts_vec again
                            p_ = np.copy(pts_vec[ct_, :])
                            p_rot = geo_utils.rotVbyW(p_, rot0axis, np.pi / 180 * (rot0ang))
                            pts_vec[ct_, :] = p_rot

                    # Temporary ref axis node coordinates - aligned with main system of reference
                    if xFraction:
                        # getting the bounds of the FFD section
                        x_min = np.min(pts_vec[:, 0])
                        x_max = np.max(pts_vec[:, 0])
                        x_node = xFraction * (x_max - x_min) + x_min  # chordwise
                    else:
                        x_node = np.mean(pts_vec[:, 0])

                    if yFraction:
                        y_min = np.min(pts_vec[:, 1])
                        y_max = np.max(pts_vec[:, 1])
                        y_node = y_max - yFraction * (y_max - y_min)  # top-bottom
                    else:
                        y_node = np.mean(pts_vec[:, 1])

                    if zFraction:
                        z_min = np.min(pts_vec[:, 2])
                        z_max = np.max(pts_vec[:, 2])
                        z_node = z_max - zFraction * (z_max - z_min)  # top-bottom
                    else:
                        z_node = np.mean(pts_vec[:, 2])

                    # This is the FFD ref axis node - if the block has not been rotated
                    nd = [x_node, y_node, z_node]
                    nd_final = np.copy(nd)

                    if rot0ang:
                        # rotating the non-aligned FFDs back in position
                        nd_final[:] = geo_utils.rotVbyW(nd, rot0axis, np.pi / 180 * (-rot0ang))

                    # insert the final coordinates in the var to be passed to pySpline:
                    refaxisNodes[place + i, 0] = nd_final[0]
                    refaxisNodes[place + i, 1] = nd_final[1]
                    refaxisNodes[place + i, 2] = nd_final[2]

                place += i + 1

            # Add additional volumes
            for iVol in includeVols:
                if iVol not in volumes:
                    volumes.append(iVol)

            # Generate reference axis pySpline curve
            curve = Curve(X=refaxisNodes, k=2)
            nAxis = len(curve.coef)
            self.axis[name] = {
                "curve": curve,
                "volumes": volumes,
                "rotType": rotType,
                "axis": axis,
                "rot0ang": rot0ang,
                "rot0axis": rot0axis,
                "rotAxisVar": rotAxisVar,
            }
        else:
            raise Error("One of 'curve' or 'xFraction' must be " "specified for a call to addRefAxis")

        # Specify indices to be ignored
        self.axis[name]["ignoreInd"] = ignoreInd

        # Add the raySize multiplication factor for this axis
        self.axis[name]["raySize"] = raySize

        # do the same for the other half if we have a symmetry plane
        if self.FFD.symmPlane is not None:
            # we need to figure out the correct indices to ignore for the mirrored FFDs

            # first get the matching indices between the current and mirroring FFDs.
            # we want to include the the nodes on the symmetry plane.
            # these will appear as the same indices on FFDs on both sides
            indSetA, indSetB = self.getSymmetricCoefList(getSymmPlane=True)

            # loop over the inds_to_ignore list and find the corresponding symmetries
            ignoreIndSymm = []
            for ind in ignoreInd:
                try:
                    tmp = indSetA.index(ind)
                except ValueError:
                    raise Error(
                        f"""The index {ind} is not in indSetA. This is likely due to a weird
                        issue caused by the point reduction routines during initialization.
                        Reduce the offset of the FFD control points from the symmetry plane
                        to avoid it. The max deviation from the symmetry plane needs to be
                        less than around 1e-5 if rest of the default tolerances in pygeo is used."""
                    )
                ind_mirror = indSetB[tmp]
                ignoreIndSymm.append(ind_mirror)

            self.axis[name + "Symm"]["ignoreInd"] = ignoreIndSymm

            # we just take the same raySize as the original curve
            self.axis[name + "Symm"]["raySize"] = raySize

        return nAxis

    def addPointSet(self, points, ptName, origConfig=True, **kwargs):
        """
        Add a set of coordinates to DVGeometry
        The is the main way that geometry, in the form of a coordinate
        list is given to DVGeometry to be manipulated.
        Parameters
        ----------
        points : array, size (N,3)
            The coordinates to embed. These coordinates *should* all
            project into the interior of the FFD volume.
        ptName : str
            A user supplied name to associate with the set of
            coordinates. This name will need to be provided when
            updating the coordinates or when getting the derivatives
            of the coordinates.
        origConfig : bool
            Flag determine if the coordinates are projected into the
            undeformed or deformed configuration. This should almost
            always be True except in circumstances when the user knows
            exactly what they are doing.
        coord_xfer : function
            A callback function that performs a coordinate transformation
            between the DVGeo reference frame and any other reference
            frame. The DVGeo object uses this routine to apply the coordinate
            transformation in "forward" and "reverse" directions to go between
            the two reference frames. Derivatives are also adjusted since they
            are vectors coming into DVGeo (in the reverse AD mode)
            and need to be rotated. We have a callback function here that lets
            the user to do whatever they want with the coordinate transformation.
            The function must have the first positional argument as the array that is
            (npt, 3) and the two keyword arguments that must be available are "mode"
            ("fwd" or "bwd") and "apply_displacement" (True or False). This function
            can then be passed to DVGeo through something like ADflow, where the
            set DVGeo call can be modified as:
            CFDSolver.setDVGeo(DVGeo, pointSetKwargs={"coord_xfer": coord_xfer})
            An example function is as follows:
            .. code-block:: python
                def coord_xfer(coords, mode="fwd", apply_displacement=True, **kwargs):
                    # given the (npt by 3) array "coords" apply the coordinate transformation.
                    # The "fwd" mode implies we go from DVGeo reference frame to the
                    # application, e.g. CFD, the "bwd" mode is the opposite;
                    # goes from the CFD reference frame back to the DVGeo reference frame.
                    # the apply_displacement flag needs to be correctly implemented
                    # by the user; the derivatives are also passed through this routine
                    # and they only need to be rotated when going between reference frames,
                    # and they should NOT be displaced.
                    # In summary, all the displacements MUST be within the if apply_displacement == True
                    # checks, otherwise the derivatives will be wrong.
                    #  Example transfer: The CFD mesh
                    # is rotated about the x-axis by 90 degrees with the right hand rule
                    # and moved 5 units below (in z) the DVGeo reference.
                    # Note that the order of these operations is important.
                    # a different rotation matrix can be created during the creation of
                    # this function. This is a simple rotation about x-axis.
                    # Multiple rotation matrices can be used; the user is completely free
                    # with whatever transformations they want to apply here.
                    rot_mat = np.array([
                        [1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0],
                    ])
                    if mode == "fwd":
                        # apply the rotation first
                        coords_new = np.dot(coords, rot_mat)
                        # then the translation
                        if apply_displacement:
                            coords_new[:, 2] -= 5
                    elif mode == "bwd":
                        # apply the operations in reverse
                        coords_new = coords.copy()
                        if apply_displacement:
                            coords_new[:, 2] += 5
                        # and the rotation. note the rotation matrix is transposed
                        # for switching the direction of rotation
                        coords_new = np.dot(coords_new, rot_mat.T)
                    return coords_new
        """

        # compNames is only needed for DVGeometryMulti, so remove it if passed
        # kwargs.pop("compNames", None)

        # save this name so that we can zero out the jacobians properly
        self.ptSetNames.append(ptName)
        self.zeroJacobians([ptName])
        self.nPts[ptName] = None

        points = np.array(points).real.astype("d")
        self.points[ptName] = points

        # # save the coordinate transformation info
        # if coord_xfer is not None:
        #     self.coord_xfer[ptName] = coord_xfer

        #     # Also apply the first coordinate transformation while adding this ptset.
        #     # The child FFDs only interact with their parent FFD, and therefore,
        #     # do not need to access the coordinate transformation routine; i.e.
        #     # all transformations are applied once during the highest level DVGeo object.
        #     points = self.coord_xfer[ptName](points, mode="bwd", apply_displacement=True)

        # self.points[ptName] = points

        # Ensure we project into the undeformed geometry
        if origConfig:
            tmpCoef = self.FFD.coef.copy()
            self.FFD.coef = self.origFFDCoef
            self.FFD._updateVolumeCoef()

        # Project the last set of points into the volume
        if self.isChild:
            coefMask = self.FFD.attachPoints(self.points[ptName], ptName, interiorOnly=True, **kwargs)
        else:
            coefMask = self.FFD.attachPoints(self.points[ptName], ptName, interiorOnly=False)

        if origConfig:
            self.FFD.coef = tmpCoef
            self.FFD._updateVolumeCoef()

        # Now embed into the children:
        for child in self.children:
            child.addPointSet(points, ptName, origConfig, **kwargs)

        self.masks[ptName] = coefMask
        self.FFD.calcdPtdCoef(ptName)
        self.updated[ptName] = False

    def addChild(self, childDVGeo):
        """Embed a child FFD into this object.
        An FFD child is a 'sub' FFD that is fully contained within
        another, parent FFD. A child FFD is also an instance of
        DVGeometry which may have its own global and/or local design
        variables. Coordinates do **not** need to be added to the
        children. The parent object will take care of that in a call
        to :func:`addPointSet()`.
        See https://github.com/mdolab/pygeo/issues/7 for a description of an
        issue with Child FFDs that you should be aware of if you are combining
        shape changes of a parent FFD with rotation or shape changes of a child FFD.
        Parameters
        ----------
        childDVGeo : instance of DVGeometry
            DVGeo object to use as a sub-FFD
        """

        # Make sure the DVGeo being added is flaged as a child:
        if childDVGeo.isChild is False:
            raise Error("Trying to add a child FFD that has NOT been " "created as a child. This operation is illegal.")

        # Extract the coef from the child FFD and ref axis and embed
        # them into the parent and compute their derivatives
        iChild = len(self.children)
        childDVGeo.iChild = iChild

        self.FFD.attachPoints(childDVGeo.FFD.coef, "child%d_coef" % (iChild))
        self.FFD.calcdPtdCoef("child%d_coef" % (iChild))

        # We must finalize the Child here since we need the ref axis
        # coefficients
        if len(childDVGeo.axis) > 0:
            childDVGeo._finalizeAxis()
            self.FFD.attachPoints(childDVGeo.refAxis.coef, "child%d_axis" % (iChild))
            self.FFD.calcdPtdCoef("child%d_axis" % (iChild))

        # Add the child to the parent and return
        self.children.append(childDVGeo)

    def addGlobalDV(self, dvName, value, func, lower=None, upper=None, scale=1.0, config=None):
        """
        Add a global design variable to the DVGeometry object. This
        type of design variable acts on one or more reference axis.
        Parameters
        ----------
        dvName : str
            A unique name to be given to this design variable group
        value : float, or iterable list of floats
            The starting value(s) for the design variable. This
            parameter may be a single variable or a numpy array
            (or list) if the function requires more than one
            variable. The number of variables is determined by the
            rank (and if rank ==1, the length) of this parameter.
        lower : float, or iterable list of floats
            The lower bound(s) for the variable(s). A single variable
            is permissable even if an array is given for value. However,
            if an array is given for ``lower``, it must be the same length
            as ``value``
        func : python function
            The python function handle that will be used to apply the
            design variable
        upper : float, or iterable list of floats
            The upper bound(s) for the variable(s). Same restrictions as
            ``lower``
        scale : float, or iterable list of floats
            The scaling of the variables. A good approximate scale to
            start with is approximately 1.0/(upper-lower). This gives
            variables that are of order ~1.0.
        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable applies to *ALL* configurations.
        """
        # if the parent DVGeometry object has a name attribute, prepend it
        if self.name is not None:
            dvName = self.name + "_" + dvName

        if isinstance(config, str):
            config = [config]
        self.DV_listGlobal[dvName] = geoDVGlobal(dvName, value, lower, upper, scale, func, config)

    def addLocalDV_Mode(self, dvName, phi, lower=None, upper=None, scale=1.0,
                      axis='y', volList=None, pointSelect=None, config=None):
        """
        Add Modes of Local FFD DVs.
        These modes are combinations of local FFD DVs.
        This function is based on 'addGeoDVLocal'


        Parameters
        ----------
        dvName : str
            A unique name to be given to this design variable group
        phi    : float, matrix
            New combinations/ Modes/ Subsapces
            
        lower : float
            The lower bound for the variable(s). This will be applied to
            all shape variables

        upper : float
            The upper bound for the variable(s). This will be applied to
            all shape variables

        scale : flot
            The scaling of the variables. A good approximate scale to
            start with is approximately 1.0/(upper-lower). This gives
            variables that are of order ~1.0.
            
        axis : str. Default is 'y'
            The coordinate directions to move. Permissible values are 'x',
            'y' and 'z'. If more than one direction is required, use multiple
            calls to addGeoDVLocal with different axis values
        volList : list
            Use the control points on the volume indicies given in volList
            
        pointSelect : pointSelect object. Default is None Use a
            pointSelect object to select a subset of the total number
            of control points. See the documentation for the
            pointSelect class in geo_utils.

        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple 
            configurations. The default value of None implies that the design
            variable appies to *ALL* configurations.

        Returns
        -------
        N : int
            The number of design variables added. 

        """
        if type(config) == str:
            config = [config]

        # Just take'em all
        #ind = np.arange(phi.shape[0])

        self.DV_listLocal[dvName] = LocalDV_Mode(dvName, lower, upper, scale, axis, phi, config)
            
        return self.DV_listLocal[dvName].nVal

    def LocalDV_NNMode(self, dvName, ndv, nFFDpts, FFDmaxy, filename, lower=None, upper=None, scale=1.0,
                      axis='y', volList=None, pointSelect=None, config=None):
        """
        Add Modes of Local FFD DVs.
        These modes are combinations of local FFD DVs.
        This function is based on 'addGeoDVLocal'


        Parameters
        ----------
        dvName : str
            A unique name to be given to this design variable group
        phi    : float, matrix
            New combinations/ Modes/ Subsapces
            
        lower : float
            The lower bound for the variable(s). This will be applied to
            all shape variables

        upper : float
            The upper bound for the variable(s). This will be applied to
            all shape variables

        scale : flot
            The scaling of the variables. A good approximate scale to
            start with is approximately 1.0/(upper-lower). This gives
            variables that are of order ~1.0.
            
        axis : str. Default is 'y'
            The coordinate directions to move. Permissible values are 'x',
            'y' and 'z'. If more than one direction is required, use multiple
            calls to addGeoDVLocal with different axis values
        volList : list
            Use the control points on the volume indicies given in volList
            
        pointSelect : pointSelect object. Default is None Use a
            pointSelect object to select a subset of the total number
            of control points. See the documentation for the
            pointSelect class in geo_utils.

        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple 
            configurations. The default value of None implies that the design
            variable appies to *ALL* configurations.

        Returns
        -------
        N : int
            The number of design variables added. 

        """
        if type(config) == str:
            config = [config]

        # Just take'em all
        #ind = np.arange(phi.shape[0])

        self.DV_listLocal[dvName] = LocalDV_NNMode(dvName, lower, upper, scale, axis, ndv, nFFDpts, FFDmaxy, filename, config)
            
        return self.DV_listLocal[dvName].nVal


    def addLocalDV(
        self, dvName, lower=None, upper=None, scale=1.0, axis="y", volList=None, pointSelect=None, config=None
    ):
        """
        Add one or more local design variables ot the DVGeometry
        object. Local variables are used for small shape modifications.
        Parameters
        ----------
        dvName : str
            A unique name to be given to this design variable group
        lower : float
            The lower bound for the variable(s). This will be applied to
            all shape variables
        upper : float
            The upper bound for the variable(s). This will be applied to
            all shape variables
        scale : float
            The scaling of the variables. A good approximate scale to
            start with is approximately 1.0/(upper-lower). This gives
            variables that are of order ~1.0.
        axis : str. Default is `y`
            The coordinate directions to move. Permissible values are `x`,
            `y` and `z`. If more than one direction is required, use multiple
            calls to :func:`addLocalDV` with different axis values.
        volList : list
            Use the control points on the volume indices given in volList.
            You should use pointSelect = None, otherwise this will not work.
        pointSelect : pointSelect object. Default is None Use a
            pointSelect object to select a subset of the total number
            of control points. See the documentation for the
            pointSelect class in geo_utils. Using pointSelect discards everything in
            volList.
        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable applies to *ALL* configurations.
        Returns
        -------
        N : int
            The number of design variables added.
        Examples
        --------
        >>> # Add all variables in FFD as local shape variables
        >>> # moving in the y direction, within +/- 1.0 units
        >>> DVGeo.addLocalDV('shape_vars', lower=-1.0, upper= 1.0, axis='y')
        >>> # As above, but moving in the x and y directions.
        >>> nVar = DVGeo.addLocalDV('shape_vars_x', lower=-1.0, upper= 1.0, axis='x')
        >>> nVar = DVGeo.addLocalDV('shape_vars_y', lower=-1.0, upper= 1.0, axis='y')
        >>> # Create a point select to use: (box from (0,0,0) to (10,0,10) with
        >>> # any point projecting into the point along 'y' axis will be selected.
        >>> PS = geo_utils.PointSelect(type = 'y', pt1=[0,0,0], pt2=[10, 0, 10])
        >>> nVar = DVGeo.addLocalDV('shape_vars', lower=-1.0, upper=1.0, pointSelect=PS)
        """
        if self.name is not None:
            dvName = self.name + "_" + dvName

        if isinstance(config, str):
            config = [config]

        if pointSelect is not None:
            if pointSelect.type != "ijkBounds":
                _, ind = pointSelect.getPoints(self.FFD.coef)
            else:
                _, ind = pointSelect.getPoints_ijk(self)
        elif volList is not None:
            if self.FFD.symmPlane is not None:
                volListTmp = []
                for vol in volList:
                    volListTmp.append(vol)
                for vol in volList:
                    volListTmp.append(vol + self.FFD.nVol // 2)
                volList = volListTmp

            volList = np.atleast_1d(volList).astype("int")
            ind = []
            for iVol in volList:
                ind.extend(self.FFD.topo.lIndex[iVol].flatten())
            ind = geo_utils.unique(ind)
        else:
            # Just take'em all
            ind = np.arange(len(self.FFD.coef))

        self.DV_listLocal[dvName] = geoDVLocal(dvName, lower, upper, scale, axis, ind, self.masks, config)

        return self.DV_listLocal[dvName].nVal

    def addLocalDV_NNMode(self, dvName, ndv, nFFDpts, FFDmaxy, filename, lower=None, upper=None, scale=1.0,
                      axis='y', volList=None, pointSelect=None, config=None):
        """
        Add Modes of Local FFD DVs.
        These modes are combinations of local FFD DVs.
        This function is based on 'addGeoDVLocal'


        Parameters
        ----------
        dvName : str
            A unique name to be given to this design variable group
        phi    : float, matrix
            New combinations/ Modes/ Subsapces
            
        lower : float
            The lower bound for the variable(s). This will be applied to
            all shape variables

        upper : float
            The upper bound for the variable(s). This will be applied to
            all shape variables

        scale : flot
            The scaling of the variables. A good approximate scale to
            start with is approximately 1.0/(upper-lower). This gives
            variables that are of order ~1.0.
            
        axis : str. Default is 'y'
            The coordinate directions to move. Permissible values are 'x',
            'y' and 'z'. If more than one direction is required, use multiple
            calls to addGeoDVLocal with different axis values
        volList : list
            Use the control points on the volume indicies given in volList
            
        pointSelect : pointSelect object. Default is None Use a
            pointSelect object to select a subset of the total number
            of control points. See the documentation for the
            pointSelect class in geo_utils.

        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple 
            configurations. The default value of None implies that the design
            variable appies to *ALL* configurations.

        Returns
        -------
        N : int
            The number of design variables added. 

        """
        if type(config) == str:
            config = [config]

        # Just take'em all
        #ind = np.arange(phi.shape[0])

        self.DV_listLocal[dvName] = LocalDV_NNMode(dvName, lower, upper, scale, axis, ndv, nFFDpts, FFDmaxy, filename, config)
            
        return self.DV_listLocal[dvName].nVal

    def getSymmetricCoefList(self, volList=None, pointSelect=None, tol=1e-8, getSymmPlane=False):
        """
        Determine the pairs of coefs that need to be constrained for symmetry.
        Parameters
        ----------
        volList : list
            Use the control points on the volume indices given in volList
        pointSelect : pointSelect object. Default is None Use a
            pointSelect object to select a subset of the total number
            of control points. See the documentation for the
            pointSelect class in geo_utils.
        tol : float
              Tolerance for ignoring nodes around the symmetry plane. These should be
              merged by the network/connectivity anyway
        getSymmPlane : bool
              If this flag is set to True, we also return the points on the symmetry plane
              for all volumes. e.g. a reduced point on the symmetry plane with the same
              indices on both volumes will show up as the same value in both arrays. This
              is useful when determining the indices to ignore when adding pointsets. The
              default behavior will not include the points exactly on the symmetry plane.
              this is more useful for adding them as linear constraints
        Returns
        -------
        indSetA : list of ints
                  One half of the coefs to be constrained
        indSetB : list of ints
                  Other half of the coefs to be constrained
        """

        if self.FFD.symmPlane is None:
            # nothing to be done
            indSetA = []
            indSetB = []
        else:
            # get the direction of the symmetry plane
            if self.FFD.symmPlane.lower() == "x":
                index = 0
            elif self.FFD.symmPlane.lower() == "y":
                index = 1
            elif self.FFD.symmPlane.lower() == "z":
                index = 2

            # get the points to be matched up
            if pointSelect is not None:
                pts, ind = pointSelect.getPoints(self.FFD.coef)
            elif volList is not None:
                volListTmp = []
                for vol in volList:
                    volListTmp.append(vol)
                for vol in volList:
                    volListTmp.append(vol + self.FFD.nVol // 2)
                volList = volListTmp

                volList = np.atleast_1d(volList).astype("int")
                ind = []
                for iVol in volList:
                    ind.extend(self.FFD.topo.lIndex[iVol].flatten())
                ind = geo_utils.unique(ind)
                pts = self.FFD.coef[ind]
            else:
                # Just take'em all
                ind = np.arange(len(self.FFD.coef))
                pts = self.FFD.coef

            # Create the base points for the KD tree search. We will take the abs
            # value of the symmetry direction, that way when we search we will get
            # back index pairs which is what we want.
            baseCoords = copy.copy(pts)
            baseCoords[:, index] = abs(baseCoords[:, index])

            # now use the baseCoords to create a KD tree
            # so we can use it to find the unique nodes
            tree = cKDTree(baseCoords)

            # Now search through the +ve half of the points, ignoring anything within
            # tol of the symmetry plane to find pairs
            indSetA = []
            indSetB = []
            for pt in pts:
                if pt[index] > tol:
                    # Now find any matching nodes within tol. there should be 2 and
                    # only 2 if the mesh is symmetric
                    Ind = tree.query_ball_point(pt, tol)  # should this be a separate tol
                    if len(Ind) == 2:
                        # check which point is on which side
                        if pts[Ind[0], index] > 0:
                            # first one is on the primary side
                            indSetA.append(Ind[0])
                            indSetB.append(Ind[1])
                        else:
                            # flip the order
                            indSetA.append(Ind[1])
                            indSetB.append(Ind[0])
                    else:
                        raise Error("more than 2 coefs found that match pt")

                elif (abs(pt[index]) < tol) and getSymmPlane:
                    # this point is on the symmetry plane
                    # if everything went right so far, this should return only one point
                    Ind = tree.query_ball_point(pt, tol)
                    if len(Ind) == 1:
                        indSetA.append(Ind[0])
                        indSetB.append(Ind[0])
                    else:
                        raise Error("more than 1 coefs found that match pt on symmetry plane")

        return indSetA, indSetB

    def setDesignVars(self, dvDict):
        """
        Standard routine for setting design variables from a design
        variable dictionary.
        Parameters
        ----------
        dvDict : dict
            Dictionary of design variables. The keys of the dictionary
            must correspond to the design variable names. Any
            additional keys in the dictionary are simply ignored.
        """

        # Coefficients must be complexifed from here on if complex
        if self.complex:
            self._finalize()
            self._complexifyCoef()

        if self.useComposite:
            dvDict = self.mapXDictToDVGeo(dvDict)

        for key in dvDict:
            if key in self.DV_listGlobal:
                vals_to_set = np.atleast_1d(dvDict[key]).astype("D")
                if len(vals_to_set) != self.DV_listGlobal[key].nVal:
                    raise Error(
                        f"Incorrect number of design variables for DV: {key}.\n"
                        + f"Expecting {self.DV_listGlobal[key].nVal} variables but received {len(vals_to_set)}"
                    )

                self.DV_listGlobal[key].value = vals_to_set

            if key in self.DV_listLocal:
                vals_to_set = np.atleast_1d(dvDict[key]).astype("D")
                if len(vals_to_set) != self.DV_listLocal[key].nVal:
                    raise Error(
                        f"Incorrect number of design variables for DV: {key}.\n"
                        + f"Expecting {self.DV_listLocal[key].nVal} variables but received {len(vals_to_set)}"
                    )
                self.DV_listLocal[key].value = vals_to_set

            # Jacobians are, in general, no longer up to date
            self.zeroJacobians(self.ptSetNames)

        # Flag all the pointSets as not being up to date:
        for pointSet in self.updated:
            self.updated[pointSet] = False

        # Now call setValues on the children. This way the
        # variables will be set on the children
        for child in self.children:
            child.setDesignVars(dvDict)

    def zeroJacobians(self, ptSetNames):
        """
        set stored jacobians to None for ptSetNames
        Parameters
        ----------
        ptSetNames : list
            list of ptSetNames to zero the jacobians.
        """
        for name in ptSetNames:
            self.JT[name] = None  # J is no longer up to date

    def getValues(self):
        """
        Generic routine to return the current set of design
        variables. Values are returned in a dictionary format
        that would be suitable for a subsequent call to :func:`setDesignVars`
        Returns
        -------
        dvDict : dict
            Dictionary of design variables
        """

        dvDict = {}
        for key in self.DV_listGlobal:
            dvDict[key] = self.DV_listGlobal[key].value

        # and now the local DVs
        for key in self.DV_listLocal:
            dvDict[key] = self.DV_listLocal[key].value

        # Now call getValues on the children. This way the
        # returned dictionary will include the variables from
        # the children
        for child in self.children:
            childdvDict = child.getValues()
            dvDict.update(childdvDict)

        if self.useComposite:
            dvDict = self.mapXDictToComp(dvDict)

        # cast DVs to real if we are in real mode
        if not self.complex:
            for key, val in dvDict.items():
                dvDict[key] = val.real

        return dvDict

    def extractCoef(self, axisID):
        """Extract the coefficients for the selected reference
        axis. This should be used only inside design variable functions"""

        axisNumber = self._getAxisNumber(axisID)
        C = np.zeros((len(self.refAxis.topo.lIndex[axisNumber]), 3), self.coef.dtype)

        C[:, 0] = np.take(self.coef[:, 0], self.refAxis.topo.lIndex[axisNumber])
        C[:, 1] = np.take(self.coef[:, 1], self.refAxis.topo.lIndex[axisNumber])
        C[:, 2] = np.take(self.coef[:, 2], self.refAxis.topo.lIndex[axisNumber])

        return C

    def extractS(self, axisID):
        """Extract the parametric positions of the control
        points. This is usually used in conjunction with extractCoef()"""
        axisNumber = self._getAxisNumber(axisID)
        return self.refAxis.curves[axisNumber].s.copy()

    def _getAxisNumber(self, axisID):
        """Get the sequential axis number from the name tag axisID"""
        try:
            return list(self.axis.keys()).index(axisID)
        except IndexError as e:
            raise Error("'The 'axisID' was invalid!") from e

    def updateCalculations(self, new_pts, isComplex, config):
        """
        The core update routine. pulled out here to eliminate duplication between update and
        update_deriv.
        """

        if self.isChild:
            # If this is a child, update the links between the ref axis and the
            # coefficients on the nested FFD now that the nested FFD has been
            # moved.
            # **Important**: this expects the FFD coef to be clean on this level,
            # meaning that the only changes to FFD.coef can be coming from
            # higher levels.

            # just use complex dtype here. we will convert to real in the end
            self.links_x = self.links_x.astype("D")

            for ipt in range(self.nPtAttach):
                base_pt = self.refAxis.curves[self.curveIDs[ipt]](self.links_s[ipt])
                self.links_x[ipt] = self.FFD.coef[self.ptAttachInd[ipt], :] - base_pt

        # Run Global Design Vars
        for key in self.DV_listGlobal:
            self.DV_listGlobal[key](self, config)

        # update the reference axis now that the new global vars have been run
        self.refAxis.coef = self.coef.copy()
        self.refAxis._updateCurveCoef()

        for ipt in range(self.nPtAttach):
            base_pt = self.refAxis.curves[self.curveIDs[ipt]](self.links_s[ipt])
            # Variables for rotType = 0 rotation + scaling
            ang = self.axis[self.curveIDNames[ipt]]["rot0ang"]
            ax_dir = self.axis[self.curveIDNames[ipt]]["rot0axis"]

            scale = self.scale[self.curveIDNames[ipt]](self.links_s[ipt])
            scale_x = self.scale_x[self.curveIDNames[ipt]](self.links_s[ipt])
            scale_y = self.scale_y[self.curveIDNames[ipt]](self.links_s[ipt])
            scale_z = self.scale_z[self.curveIDNames[ipt]](self.links_s[ipt])

            rotType = self.axis[self.curveIDNames[ipt]]["rotType"]
            if rotType == 0:
                bp_ = np.copy(base_pt)  # copy of original pointset - will not be rotated

                deriv = self.refAxis.curves[self.curveIDs[ipt]].getDerivative(self.links_s[ipt])
                deriv /= geo_utils.euclideanNorm(deriv)  # Normalize
                new_vec = -np.cross(deriv, self.links_n[ipt])
                new_vec = geo_utils.rotVbyW(new_vec, deriv, self.rot_x[
                        self.curveIDs[ipt]](self.links_s[ipt])*np.pi/180)
                new_pts[ipt] = base_pt + new_vec*scale                

            else:
                rotX = geo_utils.rotxM(self.rot_x[self.curveIDNames[ipt]](self.links_s[ipt]))
                rotY = geo_utils.rotyM(self.rot_y[self.curveIDNames[ipt]](self.links_s[ipt]))
                rotZ = geo_utils.rotzM(self.rot_z[self.curveIDNames[ipt]](self.links_s[ipt]))

                D = self.links_x[ipt]

                rotM = self._getRotMatrix(rotX, rotY, rotZ, rotType)

                # if necessary, assign rotation matrix for each ffd coef
                if self.coefRotM is not None:
                    attachedPoint = self.ptAttachInd[ipt]
                    if isComplex:
                        self.coefRotM[attachedPoint] = rotM
                    else:
                        self.coefRotM[attachedPoint] = np.real(rotM)

                D = np.dot(rotM, D)
                if rotType == 7:
                    # only apply the theta rotations in certain cases
                    deriv = self.refAxis.curves[self.curveIDs[ipt]].getDerivative(self.links_s[ipt])
                    deriv /= geo_utils.euclideanNorm(deriv)  # Normalize
                    D = geo_utils.rotVbyW(
                        D, deriv, np.pi / 180 * self.rot_theta[self.curveIDNames[ipt]](self.links_s[ipt])
                    )

                elif rotType == 8:
                    varname = self.axis[self.curveIDNames[ipt]]["rotAxisVar"]
                    slVar = self.DV_listSectionLocal[varname]
                    attachedPoint = self.ptAttachInd[ipt]
                    W = slVar.sectionTransform[slVar.sectionLink[attachedPoint]][:, 2]
                    D = geo_utils.rotVbyW(D, W, np.pi / 180 * self.rot_theta[self.curveIDNames[ipt]](self.links_s[ipt]))

                D[0] *= scale_x
                D[1] *= scale_y
                D[2] *= scale_z

                if isComplex:
                    new_pts[ipt] = base_pt + D * scale
                else:
                    new_pts[ipt] = np.real(base_pt + D * scale)

    def update(self, ptSetName, childDelta=True, config=None):
        """
        This is the main routine for returning coordinates that have
        been updated by design variables.
        Parameters
        ----------
        ptSetName : str
            Name of point-set to return. This must match ones of the
            given in an :func:`addPointSet()` call.
        childDelta : bool
            Return updates on child as a delta. The user should not
            need to ever change this parameter.
        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable applies to *ALL* configurations.
        """
        self.curPtSet = ptSetName
        # We've postponed things as long as we can...do the finalization.
        self._finalize()

        # Make sure coefficients are complex
        self._complexifyCoef()

        # Set all coef Values back to initial values
        if not self.isChild:
            self.FFD.coef = self.origFFDCoef.copy()
            self._setInitialValues()

            for iChild in range(len(self.children)):
                if len(self.children[iChild].axis) > 0:
                    self.children[iChild]._finalize()
                    refaxis_ptSetName = "child%d_axis" % (iChild)
                    if refaxis_ptSetName not in self.FFD.embeddedVolumes:
                        self.FFD.attachPoints(self.children[iChild].refAxis.coef, refaxis_ptSetName)
                        self.FFD.calcdPtdCoef("child%d_axis" % (iChild))
        else:
            for iChild in range(len(self.children)):
                if len(self.children[iChild].axis) > 0:
                    refaxis_ptSetName = "child%d_axis" % (iChild)
                    if refaxis_ptSetName not in self.FFD.embeddedVolumes:
                        raise Error(
                            f"refaxis {refaxis_ptSetName} cannot be added to child FFD after child is appended to parent"
                        )

            # Update all coef
            self.FFD._updateVolumeCoef()

            # Evaluate starting pointset
            Xstart = self.FFD.getAttachedPoints(ptSetName)

            if self.complex:
                # Now we have to propagate the complex part through Xstart
                tempCoef = self.FFD.coef.copy().astype("D")
                Xstart = Xstart.astype("D")
                imag_part = np.imag(tempCoef)
                imag_j = 1j

                dPtdCoef = self.FFD.embeddedVolumes[ptSetName].dPtdCoef
                if dPtdCoef is not None:
                    for ii in range(3):
                        Xstart[:, ii] += imag_j * dPtdCoef.dot(imag_part[:, ii])

        # Step 1: Call all the design variables IFF we have ref axis:
        if len(self.axis) > 0:
            if self.complex:
                new_pts = np.zeros((self.nPtAttach, 3), "D")
            else:
                new_pts = np.zeros((self.nPtAttach, 3), "d")

            # Apply the global design variables
            self.updateCalculations(new_pts, isComplex=self.complex, config=config)

            # Put the update FFD points in their proper place
            temp = np.real(new_pts)
            np.put(self.FFD.coef[:, 0], self.ptAttachInd, temp[:, 0])
            np.put(self.FFD.coef[:, 1], self.ptAttachInd, temp[:, 1])
            np.put(self.FFD.coef[:, 2], self.ptAttachInd, temp[:, 2])

        # Now add in the local DVs
        for key in self.DV_listLocal:
            self.DV_listLocal[key](self.FFD.coef, config)

        # Update all coef
        self.FFD._updateVolumeCoef()

        # Evaluate coordinates from the parent
        Xfinal = self.FFD.getAttachedPoints(ptSetName)

        # Propagate the complex part through the volume artificially
        if self.complex:
            # Above, we only took the real part of the coef because
            # _updateVolumeCoef gets rid of it anyway. Here, we need to include
            # the complex part because we want to propagate it through
            tempCoef = self.FFD.coef.copy().astype("D")
            if len(self.axis) > 0:
                np.put(tempCoef[:, 0], self.ptAttachInd, new_pts[:, 0])
                np.put(tempCoef[:, 1], self.ptAttachInd, new_pts[:, 1])
                np.put(tempCoef[:, 2], self.ptAttachInd, new_pts[:, 2])

            for key in self.DV_listLocal:
                self.DV_listLocal[key].updateComplex(tempCoef, config)

            Xfinal = Xfinal.astype("D")
            imag_part = np.imag(tempCoef)
            imag_j = 1j

            dPtdCoef = self.FFD.embeddedVolumes[ptSetName].dPtdCoef
            if dPtdCoef is not None:
                for ii in range(3):
                    Xfinal[:, ii] += imag_j * dPtdCoef.dot(imag_part[:, ii])

        # Now loop over the children set the FFD and refAxis control
        # points as evaluated from the parent
        for iChild in range(len(self.children)):
            child = self.children[iChild]
            child._finalize()
            self.applyToChild(iChild)

            if self.complex:
                # need to propagate the sensitivity to the children Xfinal here to do this
                # correctly
                child._complexifyCoef()
                child.FFD.coef = child.FFD.coef.astype("D")

                dXrefdCoef = self.FFD.embeddedVolumes["child%d_axis" % (iChild)].dPtdCoef
                dCcdCoef = self.FFD.embeddedVolumes["child%d_coef" % (iChild)].dPtdCoef

                if dXrefdCoef is not None:
                    for ii in range(3):
                        child.coef[:, ii] += imag_j * dXrefdCoef.dot(imag_part[:, ii])

                if dCcdCoef is not None:
                    for ii in range(3):
                        child.FFD.coef[:, ii] += imag_j * dCcdCoef.dot(imag_part[:, ii])
                child.refAxis.coef = child.coef.copy()
                child.refAxis._updateCurveCoef()

            Xfinal += child.update(ptSetName, childDelta=True, config=config)

        self._unComplexifyCoef()

        # Finally flag this pointSet as being up to date:
        self.updated[ptSetName] = True

        if self.isChild and childDelta:
            return Xfinal - Xstart
        else:
            # we only check if we need to apply the coordinate transformation
            # and move the pointset to the reference frame of the application,
            # if this is the last pygeo in the chain
            if ptSetName in self.coord_xfer:
                Xfinal = self.coord_xfer[ptSetName](Xfinal, mode="fwd", apply_displacement=True)
            return Xfinal

    def applyToChild(self, iChild):
        """
        This function is used to apply the changes in the parent FFD to the
        child FFD points and child reference axis points.
        """
        child = self.children[iChild]

        # Set FFD points and reference axis points from parent
        child.FFD.coef = self.FFD.getAttachedPoints("child%d_coef" % (iChild))
        child.coef = self.FFD.getAttachedPoints("child%d_axis" % (iChild))

        # Update the reference axes on the child
        child.refAxis.coef = child.coef.copy()
        child.refAxis._updateCurveCoef()

    def convertSensitivityToDict(self, dIdx, out1D=False, useCompositeNames=False):
        """
        This function takes the result of totalSensitivity and
        converts it to a dict for use in pyOptSparse
        Parameters
        ----------
        dIdx : array
           Flattened array of length getNDV(). Generally it comes from
           a call to totalSensitivity()
        out1D : boolean
            If true, creates a 1D array in the dictionary instead of 2D.
            This function is used in the matrix-vector product calculation.
        useCompositeNames : boolean
            Whether the sensitivity dIdx is with respect to the composite DVs or the original DVGeo DVs.
            If False, the returned dictionary will have keys corresponding to the original set of geometric DVs.
            If True,  the returned dictionary will have replace those with a single key corresponding to the composite DV name.
        Returns
        -------
        dIdxDict : dictionary
           Dictionary of the same information keyed by this object's
           design variables
        """

        # compute the various DV offsets
        DVCountGlobal, DVCountLocal = self._getDVOffsets()

        i = DVCountGlobal
        dIdxDict = {}
        for key in self.DV_listGlobal:
            dv = self.DV_listGlobal[key]
            if out1D:
                dIdxDict[dv.name] = np.ravel(dIdx[:, i : i + dv.nVal])
            else:
                dIdxDict[dv.name] = dIdx[:, i : i + dv.nVal]
            i += dv.nVal

        i = DVCountLocal
        for key in self.DV_listLocal:
            dv = self.DV_listLocal[key]
            if out1D:
                dIdxDict[dv.name] = np.ravel(dIdx[:, i : i + dv.nVal])
            else:
                dIdxDict[dv.name] = dIdx[:, i : i + dv.nVal]

            i += dv.nVal

        # Add in child portion
        for iChild in range(len(self.children)):
            childdIdx = self.children[iChild].convertSensitivityToDict(
                dIdx, out1D=out1D, useCompositeNames=useCompositeNames
            )
            # update the total sensitivities with the derivatives from the child
            for key in childdIdx:
                if key in dIdxDict.keys():
                    dIdxDict[key] += childdIdx[key]
                else:
                    dIdxDict[key] = childdIdx[key]

        # replace other names with user
        if useCompositeNames and self.useComposite:
            array = []
            for _key, val in dIdxDict.items():
                array.append(val)
            array = np.hstack(array)
            dIdxDict = {self.DVComposite.name: array}

        return dIdxDict

    def convertDictToSensitivity(self, dIdxDict):
        """
        This function performs the reverse operation of
        convertSensitivityToDict(); it transforms the dictionary back
        into an array. This function is important for the matrix-free
        interface.
        Parameters
        ----------
        dIdxDict : dictionary
           Dictionary of information keyed by this object's
           design variables
        Returns
        -------
        dIdx : array
           Flattened array of length getNDV().
        """
        DVCountGlobal, DVCountLocal = self._getDVOffsets()
        dIdx = np.zeros(self.nDV_T, self.dtype)
        i = DVCountGlobal
        for key in self.DV_listGlobal:
            dv = self.DV_listGlobal[key]
            dIdx[i : i + dv.nVal] = dIdxDict[dv.name]
            i += dv.nVal

        i = DVCountLocal
        for key in self.DV_listLocal:
            dv = self.DV_listLocal[key]
            dIdx[i : i + dv.nVal] = dIdxDict[dv.name]
            i += dv.nVal

        # Note: not sure if this works with (multiple) sibling child FFDs
        for iChild in range(len(self.children)):
            childdIdx = self.children[iChild].convertDictToSensitivity(dIdxDict)
            # update the total sensitivities with the derivatives from the child
            dIdx += childdIdx
        return dIdx

    def getVarNames(self, pyOptSparse=False):
        """
        Return a list of the design variable names. This is typically
        used when specifying a wrt= argument for pyOptSparse.
        Parameters
        ----------
        pyOptSparse : bool
            Flag to specify whether the DVs returned should be those in the optProb or those internal to DVGeo.
            Only relevant if using composite DVs.
        Examples
        --------
        optProb.addCon(.....wrt=DVGeo.getVarNames())
        """
        if not pyOptSparse or not self.useComposite:
            names = list(self.DV_listGlobal.keys())
            names.extend(list(self.DV_listLocal.keys()))
        else:
            names = [self.DVComposite.name]

        # Call the children recursively
        for iChild in range(len(self.children)):
            names.extend(self.children[iChild].getVarNames())

        return names

    def totalSensitivity(self, dIdpt, ptSetName, comm=None, config=None):
        r"""
        This function computes sensitivity information.
        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}}^T \frac{dI}{d_{pt}}`
        Parameters
        ----------
        dIdpt : array of size (Npt, 3) or (N, Npt, 3)
            This is the total derivative of the objective or function
            of interest with respect to the coordinates in
            'ptSetName'. This can be a single array of size (Npt, 3)
            **or** a group of N vectors of size (Npt, 3, N). If you
            have many to do, it is faster to do many at once.
        ptSetName : str
            The name of set of points we are dealing with
        comm : MPI.IntraComm
            The communicator to use to reduce the final derivative. If
            comm is None, no reduction takes place.
        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable applies to *ALL* configurations.
        Returns
        -------
        dIdxDict : dic
            The dictionary containing the derivatives, suitable for
            pyOptSparse
        Notes
        -----
        The ``child`` and ``nDVStore`` options are only used
        internally and should not be changed by the user.
        """

        # Make dIdpt at least 3D
        if len(dIdpt.shape) == 2:
            dIdpt = np.array([dIdpt])
        N = dIdpt.shape[0]

        # apply the coordinate transformation on dIdpt if this pointset has it.
        if ptSetName in self.coord_xfer:
            # loop over functions
            for ifunc in range(N):
                # its important to remember that dIdpt are vector-like values,
                # so we don't apply the transformations and only the rotations!
                dIdpt[ifunc] = self.coord_xfer[ptSetName](dIdpt[ifunc], mode="bwd", apply_displacement=False)

        # generate the total Jacobian self.JT
        self.computeTotalJacobian(ptSetName, config=config)

        # now that we have self.JT compute the Mat-Mat multiplication
        nDV = self._getNDV()
        dIdx_local = np.zeros((N, nDV), "d")
        for i in range(N):
            if self.JT[ptSetName] is not None:
                dIdx_local[i, :] = self.JT[ptSetName].dot(dIdpt[i, :, :].flatten())

        if comm:  # If we have a comm, globaly reduce with sum
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        if self.useComposite:
            dIdx = self.mapSensToComp(dIdx)

        # Now convert to dict:
        dIdx = self.convertSensitivityToDict(dIdx, useCompositeNames=True)

        return dIdx

    def totalSensitivityProd(self, vec, ptSetName, config=None):
        r"""
        This function computes sensitivity information.
        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}} \times\mathrm{vec}`
        This is useful for forward AD mode.
        Parameters
        ----------
        vec : dictionary whose keys are the design variable names, and whose
              values are the derivative seeds of the corresponding design variable.
        ptSetName : str
            The name of set of points we are dealing with
        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable applies to *ALL* configurations.
        Returns
        -------
        xsdot : array (Nx3) -> Array with derivative seeds of the surface nodes.
        """

        self.computeTotalJacobian(ptSetName)  # This computes and updates self.JT

        names = self.getVarNames()
        newvec = np.zeros(self.getNDV(), self.dtype)

        i = 0
        missingVars = set()  # set of variables
        for vecKey in vec:
            # check if the seed DV is actually a design variable for the DVGeo object
            if vecKey not in names:
                raise Error(f"{vecKey} is not a design variable, the full list is:{names}")

        DVGeoList = self.getFlattenedChildren()

        # iterate over parent and children/grandchildren FFDs
        for geoObj in DVGeoList:
            for key in names:
                if key in geoObj.DV_listGlobal:
                    dv = geoObj.DV_listGlobal[key]
                    missingVars.discard(key)  # remove DV from missing list, if present
                elif key in geoObj.DV_listLocal:
                    dv = geoObj.DV_listLocal[key]
                    missingVars.discard(key)
                else:
                    # keep track of DVs which are in the full name list but not in this DVGeo object
                    missingVars.add(key)
                    continue

                if key in vec:
                    newvec[i : i + dv.nVal] = vec[key]  # Update the DV vector with the seed

                i += dv.nVal  # update the starting position in the vector update for the next key

        if missingVars:
            # if a DV name is listed by getVarNames() but was not found in the previous loop then something is wrong...
            raise Error(f"The following DV did not belong to any DVGeo object: {missingVars}")

        # perform the product
        if self.JT[ptSetName] is None:
            xsdot = np.zeros((0, 3))
        else:
            xsdot = self.JT[ptSetName].T.dot(newvec)
            xsdot.reshape(len(xsdot) // 3, 3)

            # check if we have a coordinate transformation on this ptset
            if ptSetName in self.coord_xfer:
                # its important to remember that dIdpt are vector-like values,
                # so we don't apply the transformations and only the rotations!
                xsdot = self.coord_xfer[ptSetName](xsdot, mode="fwd", apply_displacement=False)

            # Maybe this should be:
            # xsdot = xsdot.reshape(len(xsdot)//3, 3)

        return xsdot

    def totalSensitivityTransProd(self, vec, ptSetName, config=None):
        r"""
        This function computes sensitivity information.
        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}}^T \times\mathrm{vec}`
        This is useful for reverse AD mode.
        Parameters
        ----------
        dIdpt : array of size (Npt, 3) or (N, Npt, 3)
            This is the total derivative of the objective or function
            of interest with respect to the coordinates in
            'ptSetName'. This can be a single array of size (Npt, 3)
            **or** a group of N vectors of size (Npt, 3, N). If you
            have many to do, it is faster to do many at once.
        ptSetName : str
            The name of set of points we are dealing with
        comm : MPI.IntraComm
            The communicator to use to reduce the final derivative. If
            comm is None, no reduction takes place.
        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable applies to *ALL* configurations.
        Returns
        -------
        dIdxDict : dic
            The dictionary containing the derivatives, suitable for
            pyOptSparse
        Notes
        -----
        The ``child`` and ``nDVStore`` options are only used
        internally and should not be changed by the user.
        """

        self.computeTotalJacobian(ptSetName, config=config)

        # perform the product
        if self.JT[ptSetName] is None:
            xsdot = np.zeros((0, 3))
        else:

            # check if we have a coordinate transformation on this ptset
            if ptSetName in self.coord_xfer:
                # its important to remember that dIdpt are vector-like values,
                # so we don't apply the transformations and only the rotations!
                vec = self.coord_xfer[ptSetName](vec, mode="bwd", apply_displacement=False)

            xsdot = self.JT[ptSetName].dot(np.ravel(vec))

        # Pack result into dictionary
        xsdict = {}
        names = self.getVarNames()
        i = 0
        for key in names:
            if key in self.DV_listGlobal:
                dv = self.DV_listGlobal[key]
            else:
                dv = self.DV_listLocal[key]
            xsdict[key] = xsdot[i : i + dv.nVal]
            i += dv.nVal

        return xsdict

    def computeDVJacobian(self, config=None):
        """
        return J_temp for a given config
        """
        # These routines are not recursive. They compute the derivatives at this level and
        # pass information down one level for the next pass call from the routine above

        # This is going to be DENSE in general
        J_attach = self._attachedPtJacobian(config=config)

        # This is the sparse jacobian for the local DVs that affect
        # Control points directly.
        J_local = self._localDVJacobian(config=config)

        # this is the jacobian from accumulated derivative dependence from parent to child
        J_casc = self._cascadedDVJacobian(config=config)

        J_temp = None

        # add them together
        if J_attach is not None:
            J_temp = sparse.lil_matrix(J_attach)

        if J_local is not None:
            if J_temp is None:
                J_temp = sparse.lil_matrix(J_local)
            else:
                J_temp += J_local

        if J_casc is not None:
            if J_temp is None:
                J_temp = sparse.lil_matrix(J_casc)
            else:
                J_temp += J_casc

        return J_temp

    def computeTotalJacobian(self, ptSetName, config=None):
        """Return the total point jacobian in CSR format since we
        need this for TACS"""

        # Finalize the object, if not done yet
        self._finalize()
        self.curPtSet = ptSetName

        if self.JT[ptSetName] is not None:
            return

        # compute the derivatives of the coefficients of this level wrt all of the design
        # variables at this level and all levels above
        J_temp = self.computeDVJacobian(config=config)

        # now get the derivative of the points for this level wrt the coefficients(dPtdCoef)
        if self.FFD.embeddedVolumes[ptSetName].dPtdCoef is not None:
            dPtdCoef = self.FFD.embeddedVolumes[ptSetName].dPtdCoef.tocoo()
            # We have a slight problem...dPtdCoef only has the shape
            # functions, so it size Npt x Coef. We need a matrix of
            # size 3*Npt x 3*nCoef, where each non-zero entry of
            # dPtdCoef is replaced by value * 3x3 Identity matrix.

            # Extract IJV Triplet from dPtdCoef
            row = dPtdCoef.row
            col = dPtdCoef.col
            data = dPtdCoef.data

            new_row = np.zeros(3 * len(row), "int")
            new_col = np.zeros(3 * len(row), "int")
            new_data = np.zeros(3 * len(row))

            # Loop over each entry and expand:
            for j in range(3):
                new_data[j::3] = data
                new_row[j::3] = row * 3 + j
                new_col[j::3] = col * 3 + j

            # Size of New Matrix:
            Nrow = dPtdCoef.shape[0] * 3
            Ncol = dPtdCoef.shape[1] * 3

            # Create new matrix in coo-dinate format and convert to csr
            new_dPtdCoef = sparse.coo_matrix((new_data, (new_row, new_col)), shape=(Nrow, Ncol)).tocsr()

            # Do Sparse Mat-Mat multiplication and resort indices
            if J_temp is not None:
                self.JT[ptSetName] = (J_temp.T * new_dPtdCoef.T).tocsr()
                self.JT[ptSetName].sort_indices()

            # Add in child portion
            for iChild in range(len(self.children)):

                # Reset control points on child for child link derivatives
                self.applyToChild(iChild)
                self.children[iChild].computeTotalJacobian(ptSetName)

                self.JT[ptSetName] = self.JT[ptSetName] + self.children[iChild].JT[ptSetName]

        else:
            self.JT[ptSetName] = None

    def computeTotalJacobianCS(self, ptSetName, config=None):
        """Return the total point jacobian in CSR format since we
        need this for TACS"""

        self._finalize()
        self.curPtSet = ptSetName

        if self.JT[ptSetName] is not None:
            return

        if self.isChild:
            refFFDCoef = copy.copy(self.FFD.coef)
            refCoef = copy.copy(self.coef)

        if self.nPts[ptSetName] is None:
            self.nPts[ptSetName] = len(self.update(ptSetName).flatten())
        for child in self.children:
            child.nPts[ptSetName] = self.nPts[ptSetName]

        DVGlobalCount, DVLocalCount = self._getDVOffsets()

        h = 1e-40j

        self.JT[ptSetName] = np.zeros([self.nDV_T, self.nPts[ptSetName]])
        self._complexifyCoef()
        for key in self.DV_listGlobal:
            for j in range(self.DV_listGlobal[key].nVal):
                if self.isChild:
                    self.FFD.coef = refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = refCoef.copy()
                    self.refAxis._updateCurveCoef()

                refVal = self.DV_listGlobal[key].value[j]

                self.DV_listGlobal[key].value[j] += h

                deriv = np.imag(self._update_deriv_new(ptSetName, config=config).flatten()) / np.imag(h)

                self.JT[ptSetName][DVGlobalCount, :] = deriv

                DVGlobalCount += 1
                self.DV_listGlobal[key].value[j] = refVal

        self._unComplexifyCoef()

        for key in self.DV_listLocal:
            for j in range(self.DV_listLocal[key].nVal):
                if self.isChild:
                    self.FFD.coef = refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = refCoef.copy()
                    self.refAxis._updateCurveCoef()

                refVal = self.DV_listLocal[key].value[j]

                self.DV_listLocal[key].value[j] += h
                deriv = np.imag(self._update_deriv_new(ptSetName, config=config).flatten()) / np.imag(h)

                self.JT[ptSetName][DVLocalCount, :] = deriv

                DVLocalCount += 1
                self.DV_listLocal[key].value[j] = refVal

        for iChild in range(len(self.children)):
            child = self.children[iChild]
            child._finalize()

            # In the updates applied previously, the FFD points on the children
            # will have been set as deltas. We need to set them as absolute
            # coordinates based on the changes in the parent before moving down
            # to the next level
            self.applyToChild(iChild)

            # Now get jacobian from child and add to parent jacobian
            child.computeTotalJacobianCS(ptSetName, config=config)
            self.JT[ptSetName] = self.JT[ptSetName] + child.JT[ptSetName]

    def addVariablesPyOpt(
        self,
        optProb,
        globalVars=True,
        localVars=True,
        sectionlocalVars=True,
        spanwiselocalVars=True,
        ignoreVars=None,
        freezeVars=None,
    ):
        """
        Add the current set of variables to the optProb object.
        Parameters
        ----------
        optProb : pyOpt_optimization class
            Optimization problem definition to which variables are added
        globalVars : bool
            Flag specifying whether global variables are to be added
        localVars : bool
            Flag specifying whether local variables are to be added
        sectionlocalVars : bool
            Flag specifying whether section local variables are to be added
        spanwiselocalVars : bool
            Flag specifying whether spanwiselocal variables are to be added
        ignoreVars : list of strings
            List of design variables the user doesn't want to use
            as optimization variables.
        freezeVars : list of string
            List of design variables the user wants to add as optimization
            variables, but to have the lower and upper bounds set at the current
            variable. This effectively eliminates the variable, but it the variable
            is still part of the optimization.
        """
        if ignoreVars is None:
            ignoreVars = set()
        if freezeVars is None:
            freezeVars = set()

        # Add design variables from the master:
        varLists = OrderedDict(
            [
                ("globalVars", self.DV_listGlobal),
                ("localVars", self.DV_listLocal),
            ]
        )

        # we add the composite DVs, and construct linear constraints that replace the existing bounds
        # then we simply return without adding any of the other DVs
        if self.useComposite:
            dv = self.DVComposite
            optProb.addVarGroup(dv.name, dv.nVal, "c", value=dv.value, lower=dv.lower, upper=dv.upper, scale=dv.scale)

            # add the linear DV constraints that replace the existing bounds!
            # Note that we assume all DVs are added here, i.e. no ignoreVars or any of the vars = False
            if len(ignoreVars) != 0:
                warnings.warn("Use of ignoreVars is incompatible with composite DVs")
            lb = {}
            ub = {}
            for lst in varLists:
                for key in varLists[lst]:
                    dv = varLists[lst][key]
                    lb[key] = dv.lower
                    ub[key] = dv.upper

            lb = self.convertDictToSensitivity(lb)
            ub = self.convertDictToSensitivity(ub)

            optProb.addConGroup(
                f"{self.DVComposite.name}_con",
                self.getNDV(),
                lower=lb,
                upper=ub,
                scale=1.0,
                linear=True,
                wrt=self.DVComposite.name,
                jac={self.DVComposite.name: self.DVComposite.u},
            )
            return

        for lst in varLists:
            if (
                lst == "globalVars"
                and globalVars
                or lst == "localVars"
                and localVars
            ):
                for key in varLists[lst]:
                    if key not in ignoreVars:
                        dv = varLists[lst][key]
                        if key not in freezeVars:
                            optProb.addVarGroup(
                                dv.name, dv.nVal, "c", value=dv.value, lower=dv.lower, upper=dv.upper, scale=dv.scale
                            )
                        else:
                            optProb.addVarGroup(
                                dv.name, dv.nVal, "c", value=dv.value, lower=dv.value, upper=dv.value, scale=dv.scale
                            )

        # Add variables from the children
        for child in self.children:
            child.addVariablesPyOpt(
                optProb, globalVars, localVars, ignoreVars, freezeVars
            )

    def writeTecplot(self, fileName, solutionTime=None):
        """Write the (deformed) current state of the FFD's to a tecplot file,
        including the children
        Parameters
        ----------
        fileName : str
           Filename for tecplot file. Should have a .dat extension
        SolutionTime : float
            Solution time to write to the file. This could be a fictitious time to
            make visualization easier in tecplot.
        """

        # Name here doesn't matter, just take the first one
        if len(self.points) > 0:
            keyToUpdate = list(self.points.keys())[0]
            self.update(keyToUpdate, childDelta=False)

        f = openTecplot(fileName, 3)
        vol_counter = 0

        # Write master volumes:
        vol_counter += self._writeVols(f, vol_counter, solutionTime)

        # Write children volumes:
        for iChild in range(len(self.children)):
            vol_counter += self.children[iChild]._writeVols(f, vol_counter)

        closeTecplot(f)
        if len(self.points) > 0:
            self.update(keyToUpdate, childDelta=True)

    def writeRefAxes(self, fileName):
        """Write the (deformed) current state of the RefAxes to a tecplot file,
        including the children
        Parameters
        ----------
        fileName : str
           Filename for tecplot file. Should have a no extension,an
           extension will be added.
        """
        # Name here doesnt matter, just take the first one
        self.update(list(self.points.keys())[0], childDelta=False)

        gFileName = fileName + "_parent.dat"
        if not len(self.axis) == 0:
            self.refAxis.writeTecplot(gFileName, orig=True, curves=True, coef=True)
        # Write children axes:
        for iChild in range(len(self.children)):
            cFileName = fileName + f"_child{iChild:03d}.dat"
            self.children[iChild].refAxis.writeTecplot(cFileName, orig=True, curves=True, coef=True)

    def writePointSet(self, name, fileName, solutionTime=None):
        """
        Write a given point set to a tecplot file
        Parameters
        ----------
        name : str
             The name of the point set to write to a file
        fileName : str
           Filename for tecplot file. Should have no extension, an
           extension will be added
        SolutionTime : float
            Solution time to write to the file. This could be a fictitious time to
            make visualization easier in tecplot.
        """

        coords = self.update(name, childDelta=False)
        fileName = fileName + "_%s.dat" % name
        f = openTecplot(fileName, 3)
        writeTecplot1D(f, name, coords, solutionTime)
        closeTecplot(f)

    def writePlot3d(self, fileName):
        """Write the (deformed) current state of the FFD object into a
        plot3D file. This file could then be used as the base-line FFD
        for a subsequent optimization. This function is not typically
        used in a regular basis, but may be useful in certain
        situaions, i.e. a sequence of optimizations
        Parameters
        ----------
        fileName : str
            Filename of the plot3D file to write. Should have a .fmt
            file extension.
        """
        self.FFD.writePlot3dCoef(fileName)

    def getLocalIndex(self, iVol, comp=None):
        """Return the local index mapping that points to the global
        coefficient list for a given volume"""
        return self.FFD.topo.lIndex[iVol].copy()

    def getFlattenedChildren(self):
        """
        Return a flattened list of all DVGeo objects in the family hierarchy.
        """
        flatChildren = [self]
        for child in self.children:
            flatChildren += child.getFlattenedChildren()

        return flatChildren

    # ----------------------------------------------------------------------
    #        THE REMAINDER OF THE FUNCTIONS NEED NOT BE CALLED BY THE USER
    # ----------------------------------------------------------------------

    def _finalizeAxis(self):
        """
        Internal function that sets up the collection of curve that
        the user has added one at a time. This will create the
        internal pyNetwork object
        """
        if len(self.axis) == 0:
            return

        curves = []
        for axis in self.axis:
            curves.append(self.axis[axis]["curve"])

        # Setup the network of reference axis curves
        self.refAxis = pyNetwork(curves)
        # These are the rotations
        self.rot_x = OrderedDict()
        self.rot_y = OrderedDict()
        self.rot_z = OrderedDict()
        self.rot_theta = OrderedDict()
        self.scale = OrderedDict()
        self.scale_x = OrderedDict()
        self.scale_y = OrderedDict()
        self.scale_z = OrderedDict()
        self.coef = self.refAxis.coef  # pointer
        self.coef0 = self.coef.copy().astype(self.dtype)

        i = 0
        for key in self.axis:
            # curves in ref axis are indexed sequentially...this is ok
            # since self.axis is an ORDERED dict
            t = self.refAxis.curves[i].t
            k = self.refAxis.curves[i].k
            N = len(self.refAxis.curves[i].coef)
            z = np.zeros((N, 1), self.dtype)
            o = np.ones((N, 1), self.dtype)
            self.rot_x[key] = Curve(t=t, k=k, coef=z.copy())
            self.rot_y[key] = Curve(t=t, k=k, coef=z.copy())
            self.rot_z[key] = Curve(t=t, k=k, coef=z.copy())
            self.rot_theta[key] = Curve(t=t, k=k, coef=z.copy())
            self.scale[key] = Curve(t=t, k=k, coef=o.copy())
            self.scale_x[key] = Curve(t=t, k=k, coef=o.copy())
            self.scale_y[key] = Curve(t=t, k=k, coef=o.copy())
            self.scale_z[key] = Curve(t=t, k=k, coef=o.copy())
            i += 1

        # Need to keep track of initail scale values
        self.scale0 = self.scale.copy()
        self.scale_x0 = self.scale_x.copy()
        self.scale_y0 = self.scale_y.copy()
        self.scale_z0 = self.scale_z.copy()
        self.rot_x0 = self.rot_x.copy()
        self.rot_y0 = self.rot_y.copy()
        self.rot_z0 = self.rot_z.copy()
        self.rot_theta0 = self.rot_theta.copy()

    def _finalize(self):
        if self.finalized:
            return
        self._finalizeAxis()
        if len(self.axis) == 0:
            self.finalized = True
            self.nPtAttachFull = len(self.FFD.coef)
            return
        # What we need to figure out is which of the control points
        # are connected to an axis, and which ones are not connected
        # to an axis.

        # Retrieve all the pointset masks
        # Retrieve all the pointset masks
        coefMask = np.zeros(len(self.FFD.coef), dtype=bool)
        for ptName in self.masks:
            if self.masks[ptName] != None:
                coefMask += self.masks[ptName] # This is boolean addition.

        self.ptAttachInd = []
        self.ptAttach = []
        curveIDs = []
        s = []
        curveID = 0
        # Loop over the axis we have:
        for key in self.axis:
            vol_list = np.atleast_1d(self.axis[key]["volumes"]).astype("intc")
            temp = []
            for iVol in vol_list:
                for i in range(self.FFD.vols[iVol].nCtlu):
                    for j in range(self.FFD.vols[iVol].nCtlv):
                        for k in range(self.FFD.vols[iVol].nCtlw):
                            ind = self.FFD.topo.lIndex[iVol][i, j, k]
                            if coefMask[ind] == False:
                                temp.append(ind)

            # Unique the values and append to the master list
            curPtAttach = geo_utils.unique(temp)
            self.ptAttachInd.extend(curPtAttach)

            curPts = self.FFD.coef.take(curPtAttach, axis=0).real
            self.ptAttach.extend(curPts)

            # Now do the projections for *just* the axis defined by my
            # key.
            if self.axis[key]["axis"] is None:
                tmpIDs, tmpS0 = self.refAxis.projectPoints(curPts, curves=[curveID])
            else:
                tmpIDs, tmpS0 = self.refAxis.projectRays(
                    curPts, self.axis[key]["axis"], curves=[curveID])

            curveIDs.extend(tmpIDs)
            s.extend(tmpS0)
            curveID += 1

        self.ptAttachFull = self.FFD.coef.copy().real
        self.nPtAttach = len(self.ptAttach)
        self.nPtAttachFull = len(self.ptAttachFull)

        self.curveIDs = curveIDs
        self.curveIDNames = []
        axisKeys = list(self.axis.keys())
        for i in range(len(curveIDs)):
            self.curveIDNames.append(axisKeys[self.curveIDs[i]])

        self.links_s = np.array(s)
        self.links_x = []
        self.links_n = []

        for i in range(self.nPtAttach):
            self.links_x.append(self.ptAttach[i] - self.refAxis.curves[self.curveIDs[i]](s[i]))
            deriv = self.refAxis.curves[self.curveIDs[i]].getDerivative(self.links_s[i])
            deriv /= geo_utils.euclideanNorm(deriv)  # Normalize
            self.links_n.append(np.cross(deriv, self.links_x[-1]))  # using the element just appended to self.links_x

        self.links_x = np.array(self.links_x)
        self.links_s = np.array(self.links_s)
        self.links_n = np.array(self.links_n)
        self.finalized = True

    def _setInitialValues(self):
        if len(self.axis) > 0:
            self.coef[:, :] = copy.deepcopy(self.coef0)
            for key in self.axis:
                self.scale[key].coef[:] = copy.deepcopy(self.scale0[key].coef)
                self.scale_x[key].coef[:] = copy.deepcopy(self.scale_x0[key].coef)
                self.scale_y[key].coef[:] = copy.deepcopy(self.scale_y0[key].coef)
                self.scale_z[key].coef[:] = copy.deepcopy(self.scale_z0[key].coef)
                self.rot_x[key].coef[:] = copy.deepcopy(self.rot_x0[key].coef)
                self.rot_y[key].coef[:] = copy.deepcopy(self.rot_y0[key].coef)
                self.rot_z[key].coef[:] = copy.deepcopy(self.rot_z0[key].coef)
                self.rot_theta[key].coef[:] = copy.deepcopy(self.rot_theta0[key].coef)

    def _getRotMatrix(self, rotX, rotY, rotZ, rotType):
        if rotType == 1:
            D = np.dot(rotZ, np.dot(rotY, rotX))
        elif rotType == 2:
            D = np.dot(rotY, np.dot(rotZ, rotX))
        elif rotType == 3:
            D = np.dot(rotX, np.dot(rotZ, rotY))
        elif rotType == 4:
            D = np.dot(rotZ, np.dot(rotX, rotY))
        elif rotType == 5:
            D = np.dot(rotY, np.dot(rotX, rotZ))
        elif rotType == 6:
            D = np.dot(rotX, np.dot(rotY, rotZ))
        elif rotType == 7:
            D = np.dot(rotY, np.dot(rotX, rotZ))
        elif rotType == 8:
            D = np.dot(rotY, np.dot(rotX, rotZ))
        return D

    def _getNDV(self):
        """Return the actual number of design variables, global + local
        + section local + spanwise local
        """
        return self._getNDVGlobal() + self._getNDVLocal()

    def getNDV(self):
        """
        Return the total number of design variables this object has.
        Returns
        -------
        nDV : int
            Total number of design variables
        """
        return self._getNDV()

    def _getNDVGlobal(self):
        """
        Get total number of global variables, inclding any children
        """
        nDV = 0
        for key in self.DV_listGlobal:
            nDV += self.DV_listGlobal[key].nVal

        for child in self.children:
            nDV += child._getNDVGlobal()

        return nDV

    def _getNDVLocal(self):
        """
        Get total number of local variables, inclding any children
        """
        nDV = 0
        for key in self.DV_listLocal:
            nDV += self.DV_listLocal[key].nVal

        for child in self.children:
            nDV += child._getNDVLocal()

        return nDV

    def _getNDVSelf(self):
        """
        Get total number of local and global variables, not including
        children
        """
        return self._getNDVGlobalSelf() + self._getNDVLocalSelf()

    def _getNDVGlobalSelf(self):
        """
        Get total number of global variables, not including
        children
        """
        nDV = 0
        for key in self.DV_listGlobal:
            nDV += self.DV_listGlobal[key].nVal

        return nDV

    def _getNDVLocalSelf(self):
        """
        Get total number of local variables, not including
        children
        """
        nDV = 0
        for key in self.DV_listLocal:
            nDV += self.DV_listLocal[key].nVal

        return nDV

    def _getDVOffsets(self):
        """
        return the global and local DV offsets for this FFD
        """

        # figure out the split between local and global Variables
        # All global vars at all levels come first
        # then spanwise, then section local vars and then local vars.
        # Parent Vars come before child Vars

        # get the global and local DV numbers on the parents if we don't have them
        if (
            self.nDV_T is None
            or self.nDVG_T is None
            or self.nDVL_T is None
        ):
            self.nDV_T = self._getNDV()
            self.nDVG_T = self._getNDVGlobal()
            self.nDVL_T = self._getNDVLocal()
            self.nDVG_count = 0
            self.nDVL_count = self.nDVG_T 

        nDVG = self._getNDVGlobalSelf()
        nDVL = self._getNDVLocalSelf()

        # Set the total number of global and local DVs into any children of this parent
        for child in self.children:
            # now get the numbers for the current parent child

            child.nDV_T = self.nDV_T
            child.nDVG_T = self.nDVG_T
            child.nDVL_T = self.nDVL_T

            child.nDVG_count = self.nDVG_count + nDVG
            child.nDVL_count = self.nDVL_count + nDVL

            # Increment the counters for the children
            nDVG += child._getNDVGlobalSelf()
            nDVL += child._getNDVLocalSelf()

        return self.nDVG_count, self.nDVL_count

    def _update_deriv(self, iDV=0, oneoverh=1.0 / 1e-40, config=None, localDV=False):

        """Copy of update function for derivative calc"""
        new_pts = np.zeros((self.nPtAttach, 3), "D")

        # Step 1: Call all the design variables IFF we have ref axis:
        if len(self.axis) > 0:

            # Recompute changes due to global dvs at current point + h
            self.updateCalculations(new_pts, isComplex=True, config=config)

            # set the forward effect of the global design vars in each child
            for iChild in range(len(self.children)):

                # get the derivative of the child axis and control points wrt the parent
                # control points
                dXrefdCoef = self.FFD.embeddedVolumes["child%d_axis" % (iChild)].dPtdCoef
                dCcdCoef = self.FFD.embeddedVolumes["child%d_coef" % (iChild)].dPtdCoef

                # create a vector with the derivative of the parent control points wrt the
                # parent global variables
                tmp = np.zeros(self.FFD.coef.shape, dtype="d")
                np.put(tmp[:, 0], self.ptAttachInd, np.imag(new_pts[:, 0]) * oneoverh)
                np.put(tmp[:, 1], self.ptAttachInd, np.imag(new_pts[:, 1]) * oneoverh)
                np.put(tmp[:, 2], self.ptAttachInd, np.imag(new_pts[:, 2]) * oneoverh)

                # create variables for the total derivative of the child axis and control
                # points wrt the parent global variables
                dXrefdXdv = np.zeros((dXrefdCoef.shape[0] * 3), "d")
                dCcdXdv = np.zeros((dCcdCoef.shape[0] * 3), "d")

                # multiply the derivative of the child axis wrt the parent control points
                # by the derivative of the parent control points wrt the parent global vars.
                # this is just chain rule
                dXrefdXdv[0::3] = dXrefdCoef.dot(tmp[:, 0])
                dXrefdXdv[1::3] = dXrefdCoef.dot(tmp[:, 1])
                dXrefdXdv[2::3] = dXrefdCoef.dot(tmp[:, 2])

                # do the same for the child control points
                dCcdXdv[0::3] = dCcdCoef.dot(tmp[:, 0])
                dCcdXdv[1::3] = dCcdCoef.dot(tmp[:, 1])
                dCcdXdv[2::3] = dCcdCoef.dot(tmp[:, 2])
                if localDV:
                    self.children[iChild].dXrefdXdvl[:, iDV] += dXrefdXdv
                    self.children[iChild].dCcdXdvl[:, iDV] += dCcdXdv
                elif self._getNDVGlobalSelf():
                    self.children[iChild].dXrefdXdvg[:, iDV] += dXrefdXdv.real
                    self.children[iChild].dCcdXdvg[:, iDV] += dCcdXdv.real
        return new_pts

    def _update_deriv_new(self, ptSetName, config=None):

        """
        A version of the update_deriv function specifically for use
        in the computeTotalJacobianCS function."""
        new_pts = np.zeros((self.nPtAttachFull, 3), "D")

        # Make sure coefficients are complex
        self._complexifyCoef()

        # Set all coef Values back to initial values
        if not self.isChild:
            self.FFD.coef = self.FFD.coef.astype("D")
            self._setInitialValues()
        else:
            # Update all coef
            self.FFD.coef = self.FFD.coef.astype("D")
            self.FFD._updateVolumeCoef()

            # Evaluate starting pointset
            Xstart = self.FFD.getAttachedPoints(ptSetName)

            # Now we have to propagate the complex part through Xstart
            tempCoef = self.FFD.coef.copy().astype("D")
            Xstart = Xstart.astype("D")
            imag_part = np.imag(tempCoef)
            imag_j = 1j

            dPtdCoef = self.FFD.embeddedVolumes[ptSetName].dPtdCoef
            if dPtdCoef is not None:
                for ii in range(3):
                    Xstart[:, ii] += imag_j * dPtdCoef.dot(imag_part[:, ii])

        # Step 1: Call all the design variables IFF we have ref axis:
        if len(self.axis) > 0:

            # Compute changes due to global design vars
            self.updateCalculations(new_pts, isComplex=True, config=config)

            # Put the update FFD points in their proper place
            np.put(self.FFD.coef[:, 0], self.ptAttachInd, new_pts[:, 0])
            np.put(self.FFD.coef[:, 1], self.ptAttachInd, new_pts[:, 1])
            np.put(self.FFD.coef[:, 2], self.ptAttachInd, new_pts[:, 2])

        for key in self.DV_listLocal:
            self.DV_listLocal[key](self.FFD.coef, config)
            self.DV_listLocal[key].updateComplex(self.FFD.coef, config)

        # Update all coef
        self.FFD._updateVolumeCoef()

        # Evaluate coordinates from the parent
        Xfinal = self.FFD.getAttachedPoints(ptSetName)

        # now project derivs through from the coef to the pts
        Xfinal = Xfinal.astype("D")
        imag_part = np.imag(self.FFD.coef)
        imag_j = 1j

        dPtdCoef = self.FFD.embeddedVolumes[ptSetName].dPtdCoef
        if dPtdCoef is not None:
            for ii in range(3):
                Xfinal[:, ii] += imag_j * dPtdCoef.dot(imag_part[:, ii])

        # now do the same for the children
        for iChild in range(len(self.children)):
            # first, update the coef. to their new locations
            child = self.children[iChild]
            child._finalize()
            self.applyToChild(iChild)

            # now cast forward the complex part of the derivative
            child._complexifyCoef()
            child.FFD.coef = child.FFD.coef.astype("D")

            dXrefdCoef = self.FFD.embeddedVolumes["child%d_axis" % (iChild)].dPtdCoef
            dCcdCoef = self.FFD.embeddedVolumes["child%d_coef" % (iChild)].dPtdCoef

            if dXrefdCoef is not None:
                for ii in range(3):
                    child.coef[:, ii] += imag_j * dXrefdCoef.dot(imag_part[:, ii])

            if dCcdCoef is not None:
                for ii in range(3):
                    child.FFD.coef[:, ii] += imag_j * dCcdCoef.dot(imag_part[:, ii])
            child.refAxis.coef = child.coef.copy()
            child.refAxis._updateCurveCoef()
            Xfinal += child._update_deriv_new(ptSetName, config=config)
            child._unComplexifyCoef()

        self.FFD.coef = self.FFD.coef.real.astype("d")

        if self.isChild:
            return Xfinal - Xstart
        else:
            return Xfinal

    def _complexifyCoef(self):
        """Convert coef to complex temporarily"""
        if len(self.axis) > 0:
            for key in self.axis:
                self.rot_x[key].coef = self.rot_x[key].coef.astype("D")
                self.rot_y[key].coef = self.rot_y[key].coef.astype("D")
                self.rot_z[key].coef = self.rot_z[key].coef.astype("D")
                self.rot_theta[key].coef = self.rot_theta[key].coef.astype("D")

                self.scale[key].coef = self.scale[key].coef.astype("D")
                self.scale_x[key].coef = self.scale_x[key].coef.astype("D")
                self.scale_y[key].coef = self.scale_y[key].coef.astype("D")
                self.scale_z[key].coef = self.scale_z[key].coef.astype("D")

            for i in range(self.refAxis.nCurve):
                self.refAxis.curves[i].coef = self.refAxis.curves[i].coef.astype("D")
            self.coef = self.coef.astype("D")

    def _unComplexifyCoef(self):
        """Convert coef back to reals"""
        if len(self.axis) > 0 and not self.complex:
            for key in self.axis:
                self.rot_x[key].coef = self.rot_x[key].coef.real.astype("d")
                self.rot_y[key].coef = self.rot_y[key].coef.real.astype("d")
                self.rot_z[key].coef = self.rot_z[key].coef.real.astype("d")
                self.rot_theta[key].coef = self.rot_theta[key].coef.real.astype("d")

                self.scale[key].coef = self.scale[key].coef.real.astype("d")
                self.scale_x[key].coef = self.scale_x[key].coef.real.astype("d")
                self.scale_y[key].coef = self.scale_y[key].coef.real.astype("d")
                self.scale_z[key].coef = self.scale_z[key].coef.real.astype("d")

            for i in range(self.refAxis.nCurve):
                self.refAxis.curves[i].coef = self.refAxis.curves[i].coef.real.astype("d")

            self.coef = self.coef.real.astype("d")

    def mapXDictToDVGeo(self, inDict):
        """
        Map a dictionary of DVs to the 'DVGeo' design, while keeping non-DVGeo DVs in place
        without modifying them
        Parameters
        ----------
        inDict : dict
            The dictionary of DVs to be mapped
        Returns
        -------
        dict
            The mapped DVs in the same dictionary format
        """
        # first make a copy so we don't modify in place
        inDict = copy.deepcopy(inDict)
        userVec = inDict.pop(self.DVComposite.name)
        outVec = self.mapVecToDVGeo(userVec)
        outDict = self.convertSensitivityToDict(outVec.reshape(1, -1), out1D=True, useCompositeNames=False)
        # now merge inDict and outDict
        for key in inDict:
            outDict[key] = inDict[key]
        return outDict

    def mapXDictToComp(self, inDict):
        """
        The inverse of :func:`mapXDictToDVGeo`, where we map the DVs to the composite space
        Parameters
        ----------
        inDict : dict
            The DVs to be mapped
        Returns
        -------
        dict
            The mapped DVs
        """
        # first make a copy so we don't modify in place
        inDict = copy.deepcopy(inDict)
        userVec = self.convertDictToSensitivity(inDict)
        outVec = self.mapVecToComp(userVec)
        outDict = self.convertSensitivityToDict(outVec.reshape(1, -1), out1D=True, useCompositeNames=True)
        return outDict

    def mapVecToDVGeo(self, inVec):
        """
        This is the vector version of :func:`mapXDictToDVGeo`, where the actual mapping is done
        Parameters
        ----------
        inVec : ndarray
            The DVs in a single 1D array
        Returns
        -------
        ndarray
            The mapped DVs in a single 1D array
        """
        inVec = inVec.reshape(self.getNDV(), -1)
        outVec = self.DVComposite.u @ inVec
        return outVec.flatten()

    def mapVecToComp(self, inVec):
        """
        This is the vector version of :func:`mapXDictToComp`, where the actual mapping is done
        Parameters
        ----------
        inVec : ndarray
            The DVs in a single 1D array
        Returns
        -------
        ndarray
            The mapped DVs in a single 1D array
        """
        inVec = inVec.reshape(self.getNDV(), -1)
        outVec = self.DVComposite.u.T @ inVec
        return outVec.flatten()

    def mapSensToComp(self, inVec):
        """
        Maps the sensitivity matrix to the composite design space
        Parameters
        ----------
        inVec : ndarray
            The sensitivities to be mapped
        Returns
        -------
        ndarray
            The mapped sensitivity matrix
        """
        outVec = inVec @ self.DVComposite.u  # this is the same as (self.DVComposite.u.T @ inVec.T).T
        return outVec

    def computeTotalJacobianFD(self, ptSetName, config=None):
        """This function takes the total derivative of an objective,
        I, with respect the points controlled on this processor using FD.
        We take the transpose prodducts and mpi_allreduce them to get the
        resulting value on each processor. Note that this function is slow
        and should eventually be replaced by an analytic version.
        """

        self._finalize()
        self.curPtSet = ptSetName

        if self.JT[ptSetName] is not None:
            return

        if self.isChild:
            refFFDCoef = copy.copy(self.FFD.coef)
            refCoef = copy.copy(self.coef)

        # Here we set childDelta as False, but it really doesn't matter
        # whether it is True or False because we take a difference
        # between coordsph and coords0, so the Xstart would be cancelled
        # out in the end.
        coords0 = self.update(ptSetName, childDelta=False, config=config).flatten()

        if self.nPts[ptSetName] is None:
            self.nPts[ptSetName] = len(coords0.flatten())
        for child in self.children:
            child.nPts[ptSetName] = self.nPts[ptSetName]

        DVGlobalCount, DVLocalCount = self._getDVOffsets()

        h = 1e-6

        self.JT[ptSetName] = np.zeros([self.nDV_T, self.nPts[ptSetName]])

        for key in self.DV_listGlobal:
            for j in range(self.DV_listGlobal[key].nVal):
                if self.isChild:
                    self.FFD.coef = refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = refCoef.copy()
                    self.refAxis._updateCurveCoef()

                refVal = self.DV_listGlobal[key].value[j]

                self.DV_listGlobal[key].value[j] += h

                coordsph = self.update(ptSetName, childDelta=False, config=config).flatten()

                deriv = (coordsph - coords0) / h
                self.JT[ptSetName][DVGlobalCount, :] = deriv

                DVGlobalCount += 1
                self.DV_listGlobal[key].value[j] = refVal

        for key in self.DV_listLocal:
            for j in range(self.DV_listLocal[key].nVal):
                if self.isChild:
                    self.FFD.coef = refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = refCoef.copy()
                    self.refAxis._updateCurveCoef()

                refVal = self.DV_listLocal[key].value[j]

                self.DV_listLocal[key].value[j] += h
                coordsph = self.update(ptSetName, childDelta=False, config=config).flatten()

                deriv = (coordsph - coords0) / h
                self.JT[ptSetName][DVLocalCount, :] = deriv

                DVLocalCount += 1
                self.DV_listLocal[key].value[j] = refVal

        for iChild in range(len(self.children)):
            child = self.children[iChild]
            child._finalize()

            # In the updates applied previously, the FFD points on the children
            # will have been set as deltas. We need to set them as absolute
            # coordinates based on the changes in the parent before moving down
            # to the next level
            self.applyToChild(iChild)

            # Now get jacobian from child and add to parent jacobian
            child.computeTotalJacobianFD(ptSetName, config=config)
            self.JT[ptSetName] = self.JT[ptSetName] + child.JT[ptSetName]

    def _attachedPtJacobian(self, config):
        """
        Compute the derivative of the the attached points
        """
        nDV = self._getNDVGlobalSelf()

        self._getDVOffsets()

        h = 1.0e-40j
        oneoverh = 1.0 / 1e-40
        # Just do a CS loop over the coef
        # First sum the actual number of globalDVs
        if nDV != 0:  # check this
            # create a jacobian the size of nPtAttached full by self.nDV_T, the total number of
            # dvs
            Jacobian = np.zeros((self.nPtAttachFull * 3, self.nDV_T))

            # Create the storage arrays for the information that must be
            # passed to the children
            for iChild in range(len(self.children)):
                N = self.FFD.embeddedVolumes["child%d_axis" % (iChild)].N
                # Derivative of reference axis points wrt global DVs at this level
                self.children[iChild].dXrefdXdvg = np.zeros((N * 3, self.nDV_T))

                N = self.FFD.embeddedVolumes["child%d_coef" % (iChild)].N
                # derivative of the control points wrt the global DVs at this level
                self.children[iChild].dCcdXdvg = np.zeros((N * 3, self.nDV_T))

            # We need to save the reference state so that we can always start
            # from the same place when calling _update_deriv
            if not self.isChild:
                refFFDCoef = copy.copy(self.origFFDCoef.astype("D"))
                refCoef = copy.copy(self.coef0.astype("D"))
            else:
                refFFDCoef = copy.copy(self.FFD.coef)
                refCoef = copy.copy(self.coef)

            iDV = self.nDVG_count
            for key in self.DV_listGlobal:
                if (
                    self.DV_listGlobal[key].config is None
                    or config is None
                    or any(c0 == config for c0 in self.DV_listGlobal[key].config)
                ):
                    nVal = self.DV_listGlobal[key].nVal
                    for j in range(nVal):

                        refVal = self.DV_listGlobal[key].value[j]

                        self.DV_listGlobal[key].value[j] += h

                        # Reset coefficients
                        self.FFD.coef = refFFDCoef.astype("D")  # ffd coefficients
                        self.coef = refCoef.astype("D")
                        self.refAxis.coef = refCoef.astype("D")
                        self._complexifyCoef()  # Make sure coefficients are complex
                        self.refAxis._updateCurveCoef()

                        deriv = oneoverh * np.imag(self._update_deriv(iDV, oneoverh, config=config)).flatten()
                        # reset the FFD and axis
                        self._unComplexifyCoef()
                        self.FFD.coef = self.FFD.coef.real.astype("d")

                        np.put(Jacobian[0::3, iDV], self.ptAttachInd, deriv[0::3])
                        np.put(Jacobian[1::3, iDV], self.ptAttachInd, deriv[1::3])
                        np.put(Jacobian[2::3, iDV], self.ptAttachInd, deriv[2::3])

                        iDV += 1

                        self.DV_listGlobal[key].value[j] = refVal
                else:
                    iDV += self.DV_listGlobal[key].nVal
        else:
            Jacobian = None

        return Jacobian

    def _localDVJacobian(self, config=None):
        """
        Return the derivative of the coefficients wrt the local design
        variables
        """

        # This is relatively straight forward, since the matrix is
        # entirely one's or zeros
        nDV = self._getNDVLocalSelf()
        self._getDVOffsets()

        if nDV != 0:
            Jacobian = sparse.lil_matrix((self.nPtAttachFull * 3, self.nDV_T))

            # Create the storage arrays for the information that must be
            # passed to the children
            for iChild in range(len(self.children)):
                N = self.FFD.embeddedVolumes["child%d_axis" % (iChild)].N
                self.children[iChild].dXrefdXdvl = np.zeros((N * 3, self.nDV_T))

                N = self.FFD.embeddedVolumes["child%d_coef" % (iChild)].N
                self.children[iChild].dCcdXdvl = np.zeros((N * 3, self.nDV_T))

            iDVLocal = self.nDVL_count
            for key in self.DV_listLocal:
                if self.DV_listLocal[key].config is None or \
                   config in self.DV_listLocal[key].config:

                    self.DV_listLocal[key](self.FFD.coef, config)

                    nVal = self.DV_listLocal[key].nVal

                    if self.DV_listLocal[key].localmode:
                        # this is DVlocal_mode, and I'm sure he has no children.
                        thisphi = self.DV_listLocal[key].phi
                        for j in range(nVal):
                            for i in range(thisphi.shape[1]):
                                irow = i*3 + 1
                                Jacobian[irow, iDVLocal] = thisphi[j,i]
                            iDVLocal += 1                                
                    else:
                        for j in range(nVal):
                            pt_dv = self.DV_listLocal[key].coefList[j]
                            irow = pt_dv[0] * 3 + pt_dv[1]
                            Jacobian[irow, iDVLocal] = 1.0

                            for iChild in range(len(self.children)):
                                # Get derivatives of child ref axis and FFD control
                                # points w.r.t. parent's FFD control points
                                dXrefdCoef = self.FFD.embeddedVolumes["child%d_axis" % (iChild)].dPtdCoef
                                dCcdCoef = self.FFD.embeddedVolumes["child%d_coef" % (iChild)].dPtdCoef

                                tmp = np.zeros(self.FFD.coef.shape, dtype="d")

                                tmp[pt_dv[0], pt_dv[1]] = 1.0

                                dXrefdXdvl = np.zeros((dXrefdCoef.shape[0] * 3), "d")
                                dCcdXdvl = np.zeros((dCcdCoef.shape[0] * 3), "d")

                                dXrefdXdvl[0::3] = dXrefdCoef.dot(tmp[:, 0])
                                dXrefdXdvl[1::3] = dXrefdCoef.dot(tmp[:, 1])
                                dXrefdXdvl[2::3] = dXrefdCoef.dot(tmp[:, 2])

                                dCcdXdvl[0::3] = dCcdCoef.dot(tmp[:, 0])
                                dCcdXdvl[1::3] = dCcdCoef.dot(tmp[:, 1])
                                dCcdXdvl[2::3] = dCcdCoef.dot(tmp[:, 2])

                                # TODO: the += here is to allow recursion check this with multiple nesting
                                # levels
                                self.children[iChild].dXrefdXdvl[:, iDVLocal] += dXrefdXdvl
                                self.children[iChild].dCcdXdvl[:, iDVLocal] += dCcdXdvl
                            iDVLocal += 1
                else:
                    iDVLocal += self.DV_listLocal[key].nVal

                # end if config check
            # end for
        else:
            Jacobian = None

        return Jacobian

    def _cascadedDVJacobian(self, config=None):
        """
        Compute the cascading derivatives from the parent to the child
        """

        if not self.isChild:
            return None

        # we are now on a child. Add in dependence passed from parent
        Jacobian = sparse.lil_matrix((self.nPtAttachFull * 3, self.nDV_T))

        # Save reference values (these are necessary so that we always start
        # from the base state on the current DVGeo, and then apply the design
        # variables from there).
        refFFDCoef = copy.copy(self.FFD.coef)
        refCoef = copy.copy(self.coef)

        h = 1.0e-40j
        oneoverh = 1.0 / 1e-40
        if self.dXrefdXdvg is not None:
            for iDV in range(self.dXrefdXdvg.shape[1]):
                nz1 = np.count_nonzero(self.dXrefdXdvg[:, iDV])
                nz2 = np.count_nonzero(self.dCcdXdvg[:, iDV])
                if nz1 + nz2 == 0:
                    continue

                # Complexify all of the coefficients
                self.FFD.coef = refFFDCoef.astype("D")
                self.coef = refCoef.astype("D")
                self._complexifyCoef()

                # Add a complex pertubation representing the change in the child
                # reference axis wrt the parent global DVs
                self.coef[:, 0] += self.dXrefdXdvg[0::3, iDV] * h
                self.coef[:, 1] += self.dXrefdXdvg[1::3, iDV] * h
                self.coef[:, 2] += self.dXrefdXdvg[2::3, iDV] * h

                # insert the new coef into the refAxis
                self.refAxis.coef = self.coef.copy()
                self.refAxis._updateCurveCoef()

                # Complexify the child FFD coords
                tmp1 = np.zeros_like(self.FFD.coef, dtype="D")

                # add the effect of the global coordinates on the actual control points
                tmp1[:, 0] = self.dCcdXdvg[0::3, iDV] * h
                tmp1[:, 1] = self.dCcdXdvg[1::3, iDV] * h
                tmp1[:, 2] = self.dCcdXdvg[2::3, iDV] * h

                self.FFD.coef += tmp1

                # Store the original FFD coordinates so that we can get the delta
                oldCoefLocations = self.FFD.coef.copy()

                # compute the deriv of the child FFD coords wrt the parent by processing
                # the above CS perturbation
                new_pts = self._update_deriv(iDV, oneoverh, config=config)

                # insert this result in the the correct locations of a vector the correct
                # size
                np.put(self.FFD.coef[:, 0], self.ptAttachInd, new_pts[:, 0])
                np.put(self.FFD.coef[:, 1], self.ptAttachInd, new_pts[:, 1])
                np.put(self.FFD.coef[:, 2], self.ptAttachInd, new_pts[:, 2])

                # We have to subtract off the oldCoefLocations because we only
                # want the cascading effect on the current design variables. The
                # complex part on oldCoefLocations was already accounted for on
                # the parent.
                self.FFD.coef -= oldCoefLocations

                # sum up all of the various influences
                Jacobian[0::3, iDV] += oneoverh * np.imag(self.FFD.coef[:, 0:1])
                Jacobian[1::3, iDV] += oneoverh * np.imag(self.FFD.coef[:, 1:2])
                Jacobian[2::3, iDV] += oneoverh * np.imag(self.FFD.coef[:, 2:3])

                # decomplexify the coefficients
                self.coef = self.coef.real.astype("d")
                self.FFD.coef = self.FFD.coef.real.astype("d")
                self._unComplexifyCoef()

        if self.dXrefdXdvl is not None:
            # Now repeat for the local variables
            for iDV in range(self.dXrefdXdvl.shape[1]):
                # check if there is any dependence on this DV
                nz1 = np.count_nonzero(self.dXrefdXdvl[:, iDV])
                nz2 = np.count_nonzero(self.dCcdXdvl[:, iDV])
                if nz1 + nz2 == 0:
                    continue

                # Complexify all of the coefficients
                self.FFD.coef = refFFDCoef.astype("D")
                self.coef = refCoef.astype("D")
                self._complexifyCoef()

                # Add a complex pertubation representing the change in the child
                # reference axis wrt the parent local DVs
                self.coef[:, 0] += self.dXrefdXdvl[0::3, iDV] * h
                self.coef[:, 1] += self.dXrefdXdvl[1::3, iDV] * h
                self.coef[:, 2] += self.dXrefdXdvl[2::3, iDV] * h

                # insert the new coef into the refAxis
                self.refAxis.coef = self.coef.copy()
                self.refAxis._updateCurveCoef()

                # Complexify the child FFD coords
                tmp1 = np.zeros_like(self.FFD.coef, dtype="D")

                # add the effect of the global coordinates on the actual control points
                tmp1[:, 0] = self.dCcdXdvl[0::3, iDV] * h
                tmp1[:, 1] = self.dCcdXdvl[1::3, iDV] * h
                tmp1[:, 2] = self.dCcdXdvl[2::3, iDV] * h

                self.FFD.coef += tmp1

                # Store the original FFD coordinates so that we can get the delta
                oldCoefLocations = self.FFD.coef.copy()

                # compute the deriv of the child FFD coords wrt the parent by processing
                # the above CS perturbation
                new_pts = self._update_deriv(iDV, oneoverh, config=config, localDV=True)
                np.put(self.FFD.coef[:, 0], self.ptAttachInd, new_pts[:, 0])
                np.put(self.FFD.coef[:, 1], self.ptAttachInd, new_pts[:, 1])
                np.put(self.FFD.coef[:, 2], self.ptAttachInd, new_pts[:, 2])

                # We have to subtract off the oldCoefLocations because we only
                # want the cascading effect on the current design variables. The
                # complex part on oldCoefLocations was already accounted for on
                # the parent.
                self.FFD.coef -= oldCoefLocations

                # sum up all of the various influences
                Jacobian[0::3, iDV] += oneoverh * np.imag(self.FFD.coef[:, 0:1])
                Jacobian[1::3, iDV] += oneoverh * np.imag(self.FFD.coef[:, 1:2])
                Jacobian[2::3, iDV] += oneoverh * np.imag(self.FFD.coef[:, 2:3])

                # decomplexify the coefficients
                self.coef = self.coef.real.astype("d")
                self.FFD.coef = self.FFD.coef.real.astype("d")
                self._unComplexifyCoef()

        return Jacobian

    def _writeVols(self, handle, vol_counter, solutionTime):
        for i in range(len(self.FFD.vols)):
            writeTecplot3D(handle, "FFD_vol%d" % i, self.FFD.vols[i].coef, solutionTime)
            self.FFD.vols[i].computeData(recompute=True)
            writeTecplot3D(handle, "embedding_vol", self.FFD.vols[i].data, solutionTime)
            vol_counter += 1

        return vol_counter

    def checkDerivatives(self, ptSetName):
        """
        Run a brute force FD check on ALL design variables
        Parameters
        ----------
        ptSetName : str
            name of the point set to check
        """

        print("Computing Analytic Jacobian...")
        self.zeroJacobians(ptSetName)
        for child in self.children:
            child.zeroJacobians(ptSetName)

        self.computeTotalJacobian(ptSetName)
        self.computeTotalJacobian_fast(ptSetName)

        Jac = copy.deepcopy(self.JT[ptSetName])

        # Global Variables
        print("========================================")
        print("             Global Variables           ")
        print("========================================")

        if self.isChild:
            refFFDCoef = copy.copy(self.FFD.coef)
            refCoef = copy.copy(self.coef)

        coords0 = self.update(ptSetName).flatten()

        h = 1e-6

        # figure out the split between local and global Variables
        DVCountGlob, DVCountLoc = self._getDVOffsets()

        for key in self.DV_listGlobal:
            for j in range(self.DV_listGlobal[key].nVal):

                print("========================================")
                print("      GlobalVar(%s), Value(%d)" % (key, j))
                print("========================================")

                if self.isChild:
                    self.FFD.coef = refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = self.coef.copy()
                    self.refAxis._updateCurveCoef()

                refVal = self.DV_listGlobal[key].value[j]

                self.DV_listGlobal[key].value[j] += h

                coordsph = self.update(ptSetName).flatten()

                deriv = (coordsph - coords0) / h

                for ii in range(len(deriv)):

                    relErr = (deriv[ii] - Jac[DVCountGlob, ii]) / (1e-16 + Jac[DVCountGlob, ii])
                    absErr = deriv[ii] - Jac[DVCountGlob, ii]

                    if abs(relErr) > h * 10 and abs(absErr) > h * 10:
                        print(ii, deriv[ii], Jac[DVCountGlob, ii], relErr, absErr)

                DVCountGlob += 1
                self.DV_listGlobal[key].value[j] = refVal

        for key in self.DV_listLocal:
            for j in range(self.DV_listLocal[key].nVal):

                print("========================================")
                print("      LocalVar(%s), Value(%d)           " % (key, j))
                print("========================================")

                if self.isChild:
                    self.FFD.coef = refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = self.coef.copy()
                    self.refAxis._updateCurveCoef()

                refVal = self.DV_listLocal[key].value[j]

                self.DV_listLocal[key].value[j] += h
                coordsph = self.update(ptSetName).flatten()

                deriv = (coordsph - coords0) / h

                for ii in range(len(deriv)):
                    relErr = (deriv[ii] - Jac[DVCountLoc, ii]) / (1e-16 + Jac[DVCountLoc, ii])
                    absErr = deriv[ii] - Jac[DVCountLoc, ii]

                    if abs(relErr) > h and abs(absErr) > h:
                        print(ii, deriv[ii], Jac[DVCountLoc, ii], relErr, absErr)

                DVCountLoc += 1
                self.DV_listLocal[key].value[j] = refVal

        for child in self.children:
            child.checkDerivatives(ptSetName)

    def printDesignVariables(self):
        """
        Print a formatted list of design variables to the screen
        """
        for dg in self.DV_listGlobal:
            print("%s" % (self.DV_listGlobal[dg].name))
            for i in range(self.DV_listGlobal[dg].nVal):
                print("%20.15f" % (self.DV_listGlobal[dg].value[i]))

        for dl in self.DV_listLocal:
            print("%s" % (self.DV_listLocal[dl].name))
            for i in range(self.DV_listLocal[dl].nVal):
                print("%20.15f" % (self.DV_listLocal[dl].value[i]))

        for child in self.children:
            child.printDesignVariables()

class geoDV:
    def __init__(self, name, value, nVal, lower, upper, scale):
        self.name = name
        self.value = value
        self.nVal = nVal
        self.lower = self.upper = self.scale = None

        if lower is not None:
            self.lower = convertTo1D(lower, self.nVal)
        if upper is not None:
            self.upper = convertTo1D(upper, self.nVal)
        if scale is not None:
            self.scale = convertTo1D(scale, self.nVal)


class geoDVGlobal(geoDV):
    def __init__(self, name, value, lower, upper, scale, function, config):
        """
        Create a geometric design variable (or design variable group)
        See addGlobalDV in DVGeometry class for more information
        """
        value = np.atleast_1d(np.array(value)).astype("D")
        super().__init__(
            name=name,
            value=value,
            nVal=len(value),
            lower=lower,
            upper=upper,
            scale=scale,
        )

        self.config = config
        self.function = function

    def __call__(self, geo, config):
        """When the object is called, actually apply the function"""
        # Run the user-supplied function
        d = np.dtype(complex)

        if self.config is None or config is None or any(c0 == config for c0 in self.config):
            # If the geo object is complex, which is indicated by .coef
            # being complex, run with complex numbers. Otherwise, convert
            # to real before calling. This eliminates casting warnings.
            if geo.coef.dtype == d or geo.complex:
                return self.function(self.value, geo)
            else:
                return self.function(np.real(self.value), geo)




class LocalDV_Mode(object):
     
    def __init__(self, dvName, lower, upper, scale, axis, phi,config):
        
        """Create a set of geometric design variables which change the shape
        of a surface surface_id. Local design variables change the surface
        in all three axis.
        See addGeoDVLocal for more information
        """
        N = 1#len(axis)
        self.nVal = phi.shape[0]*N
        self.value = np.zeros(self.nVal, 'D')
        self.name = dvName
        self.lower = None
        self.upper = None
        self.config = config
        self.phi = phi
        self.localmode = True 
        if lower is not None:
            self.lower = convertTo1D(lower, self.nVal)
        if upper is not None:
            self.upper = convertTo1D(upper, self.nVal)
        if scale is not None:
            self.scale = convertTo1D(scale, self.nVal)
        
        if axis == 'y':
            self.iy = 1
        elif axis == 'z':
            self.iy = 2   
        
        #self.coefList = np.zeros((npts, 2), 'intc')

        #j = 0
        #for i in range(npts):
            #self.coefList[j] = [coefList[i], 1]
            #j += 1
            #if 'x' in axis.lower():
            #    self.coefList[j] = [coefList[i], 0]
            #    j += 1
            #elif 'y' in axis.lower():
            #elif 'z' in axis.lower():
            #    self.coefList[j] = [coefList[i], 2]
            #    j += 1
   
    def __call__(self, coef, config):
        """When the object is called, apply the design variable values to 
        coefficients"""
        myvalue = np.zeros(self.phi.shape[1])
        for i in range(self.phi.shape[1]):
            for j in range(self.nVal):
                myvalue[i] += self.phi[j,i]*self.value[j].real
        
        if self.config is None or config in self.config:
            for i in range(self.phi.shape[1]):
                coef[i,self.iy] += myvalue[i]
      
        return coef

    def updateComplex(self, coef, config):

        myvalue = np.zeros(self.phi.shape[1])
        for i in range(self.phi.shape[1]):
            for j in range(self.nVal):
                myvalue[i] += self.phi[j,i]*self.value[j].imag
        
        if self.config is None or config in self.config:
            for i in range(self.phi.shape[1]):
                coef[i,self.iy] += myvalue[i]*1j
        return coef

class LocalDV_NNMode(object):
     
    def __init__(self, dvName, lower, upper, scale, axis, ndv, nFFDpts, FFDmaxy, filename,config):
        
        """Create a set of geometric design variables which change the shape
        of a surface surface_id. Local design variables change the surface
        in all three axis.
        See addGeoDVLocal for more information
        """
        N = 1#len(axis)
        self.nVal = ndv*N
        self.nFFDpts = nFFDpts
        self.FFDmaxy = FFDmaxy
        self.value = np.zeros(self.nVal, 'D')
        self.name = dvName
        self.lower = None
        self.upper = None
        self.config = config

        from keras.models import load_model
        if MPI.COMM_WORLD.rank == 0:
            self.NNmode = load_model(filename)

        self.localmode = True 
        if lower is not None:
            self.lower = convertTo1D(lower, self.nVal)
        if upper is not None:
            self.upper = convertTo1D(upper, self.nVal)
        if scale is not None:
            self.scale = convertTo1D(scale, self.nVal)
        
        if axis == 'y':
            self.iy = 1
        elif axis == 'z':
            self.iy = 2   
        
        #self.coefList = np.zeros((npts, 2), 'intc')

        #j = 0
        #for i in range(npts):
            #self.coefList[j] = [coefList[i], 1]
            #j += 1
            #if 'x' in axis.lower():
            #    self.coefList[j] = [coefList[i], 0]
            #    j += 1
            #elif 'y' in axis.lower():
            #elif 'z' in axis.lower():
            #    self.coefList[j] = [coefList[i], 2]
            #    j += 1

    def computeFFDs_real(self):
        myinput = np.zeros((1,self.nVal))
        for i in range(self.nVal):
            myinput[i] = self.value[i].real
        
        if MPI.COMM_WORLD.rank == 0:
            tempscores = self.NNmode.predict(myinput)
            scores = np.zeros(self.nFFDpts)
            scores[:] = tempscores[0,:]*self.FFDmaxy
        else:
            scores = np.zeros(self.nFFDpts)
        MPI.COMM_WORLD.Bcast(scores, root=0)
        return scores       

    def computeFFDs_imag(self):
        myinput = np.zeros((1,self.nVal))
        for i in range(self.nVal):
            myinput[i] = self.value[i].imag
        
        if MPI.COMM_WORLD.rank == 0:
            tempscores = self.NNmode.predict(myinput)
            scores = np.zeros(self.nFFDpts)
            scores[:] = tempscores[0,:]*self.FFDmaxy
        else:
            scores = np.zeros(self.nFFDpts)
        MPI.COMM_WORLD.Bcast(scores, root=0)
        return scores      
   
    def __call__(self, coef, config):
        """When the object is called, apply the design variable values to 
        coefficients"""
        myvalue = self.computeFFDs_real()
        
        if self.config is None or config in self.config:
            for i in range(self.phi.shape[1]):
                coef[i,self.iy] += myvalue[i]
        return coef

    def updateComplex(self, coef, config):

        myvalue = self.computeFFDs_imag()
        
        if self.config is None or config in self.config:
            for i in range(self.phi.shape[1]):
                coef[i,self.iy] += myvalue[i]*1j
        return coef


class geoDVLocal(geoDV):
    def __init__(self, name, lower, upper, scale, axis, coefListIn, mask, config):
        """
        Create a set of geometric design variables which change the shape
        of a surface surface_id. Local design variables change the surface
        in all three axis.
        See addLocalDV for more information
        """

        coefList = []
        # create a new coefficient list that excludes any values that are masked
        for i in range(len(coefListIn)):
            if not mask[coefListIn[i]]:
                coefList.append(coefListIn[i])

        N = len(axis)
        nVal = len(coefList) * N
        super().__init__(name=name, value=np.zeros(nVal, "D"), nVal=nVal, lower=lower, upper=upper, scale=scale)

        self.config = config
        self.coefList = np.zeros((self.nVal, 2), "intc")
        j = 0

        for i in range(len(coefList)):
            if "x" in axis.lower():
                self.coefList[j] = [coefList[i], 0]
                j += 1
            elif "y" in axis.lower():
                self.coefList[j] = [coefList[i], 1]
                j += 1
            elif "z" in axis.lower():
                self.coefList[j] = [coefList[i], 2]
                j += 1

    def __call__(self, coef, config):
        """When the object is called, apply the design variable values to
        coefficients"""
        if self.config is None or config is None or any(c0 == config for c0 in self.config):
            for i in range(self.nVal):
                coef[self.coefList[i, 0], self.coefList[i, 1]] += self.value[i].real

        return coef

    def updateComplex(self, coef, config):
        if self.config is None or config is None or any(c0 == config for c0 in self.config):
            for i in range(self.nVal):
                coef[self.coefList[i, 0], self.coefList[i, 1]] += self.value[i].imag * 1j

        return coef

    def mapIndexSets(self, indSetA, indSetB):
        """
        Map the index sets from the full coefficient indices to the local set.
        """
        # Temp is the list of FFD coefficients that are included
        # as shape variables in this localDV "key"
        temp = self.coefList
        cons = []
        for j in range(len(indSetA)):
            # Try to find this index # in the coefList (temp)
            up = None
            down = None

            # Note: We are doing inefficient double looping here
            for k in range(len(temp)):
                if temp[k][0] == indSetA[j]:
                    up = k
                if temp[k][0] == indSetB[j]:
                    down = k

            # If we haven't found up AND down do nothing
            if up is not None and down is not None:
                cons.append([up, down])

        return cons



def convertTo1D(value, dim1):
    """
    Generic function to process 'value'. In the end, it must be
    array of size dim1. value is already that shape, excellent,
    otherwise, a scalar will be 'upcast' to that size
    """

    if np.isscalar:
        return value*np.ones(dim1)
    else:
        temp = np.atleast_1d(value)
        if temp.shape[0] == dim1:
            return value
        else:
            raise Error('The size of the 1D array was the incorret shape')