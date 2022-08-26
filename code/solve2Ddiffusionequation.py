###############################################################################
#
#    solve2DDiffusionEquation.py
#
#    - A script to set up and solve the 2D diffusion equation for conduction
#    in a plate, with fixed temperature boundary conditions applied on the
#    sides of the plate
#
#    [Chapter 2]
#
#    Author: Dr. Aidan Wimshurst (FluidMechanics101)
#
#    - Email: FluidMechanics101@gmail.com
#    - Web  : https://www.fluidmechanics101.com
#    - YouTube: Fluid Mechanics 101
#
#    Version 2.0.0 23/07/2019
#
#    Version 2.0.3 15/04/2022
#    - Updated for python 2 & python 3 compatability.
#
###############################################################################
#
# Required Modules for Matrices and Plotting
#
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


###############################################################################
#
#    User Inputs
#
###############################################################################
def diffusion_eqn(Nx=10, Ny=10, T_left=0, T_right=0, T_top=0, T_bottom=0):
    # Thermal Conductivity of the plate (W/mK)
    conductivity = 384

    # Thickness of the plate (m)
    thickness = 0.1

    # Length of the plate (y)
    plateLength = 1.0

    # Width of the plate (x)
    plateWidth = 1.0

    # Number of cells along the length (Nx)
    numCellsLength = Nx

    # Number of cells along the width (Ny)
    numCellsWidth = Ny

    # Temperature at the left end of the plate
    temperatureLeft = T_left

    # Temperature at the bottom of the plate
    temperatureBottom = T_bottom

    # Temperature at the right of the plate
    temperatureRight = T_right

    # Temperature at the top of the plate
    temperatureTop = T_top

    # Heat source per unit volume (W/m3)
    heatSourcePerVol = 1000

    # Plot the data?
    plotOutput = 'true'

    # Print the set up data? (table of coefficients and matrices)
    printSetup = 'false'

    # Print the solution output (the temperature vector)
    printSolution = 'true'

    # ============================================================================
    # Code Begins Here
    # ============================================================================

    # Define a line for printing (to ensure they have the same length)
    lineSingle = '------------------------------------------------'
    lineDouble = '================================================'

    print(lineDouble)
    print('')
    print('   solve2DDiffusionEquation.py')
    print('')
    print(' - Fluid Mechanics 101')
    print(' - Author: Dr. Aidan Wimshurst')
    print(' - Contact: FluidMechanics101@gmail.com')
    print('')
    print(' [Chapter 2]')
    print(lineDouble)

    ###############################################################################
    #
    #     Create the mesh of cells
    #
    ###############################################################################

    print(' Creating Mesh')
    print(lineSingle)

    # Start by calculating the number of cells
    numCells = numCellsLength * numCellsWidth

    # Calculate the number of faces
    numFaces = (numCellsLength + 1) * (numCellsWidth + 1)

    # Coodinates of the faces
    # - Assemble as a single long vector, rather than a grid.
    # - np.tile and np.repeat are useful for repeating regular patterns.
    xFacesPattern = np.linspace(0, plateLength, numCellsLength + 1)
    yFacesPattern = np.linspace(0, plateWidth, numCellsWidth + 1)
    xFaces = np.tile(xFacesPattern, numCellsWidth + 1)
    yFaces = np.repeat(yFacesPattern, numCellsLength + 1)

    # Coordinates of the centroids
    xCentroidPattern = 0.5 * (xFacesPattern[1:] + xFacesPattern[0:-1])
    yCentroidPattern = 0.5 * (yFacesPattern[1:] + yFacesPattern[0:-1])
    xCentroids = np.tile(xCentroidPattern, numCellsWidth)
    yCentroids = np.repeat(yCentroidPattern, numCellsLength)

    # Distance between the cell centroids and the boundaries
    dLeftBoundary = 2.0 * (xCentroids[0] - xFacesPattern[0])
    dTopBoundary = 2.0 * (yCentroids[0] - yFacesPattern[0])
    dRightBoundary = 2.0 * (xFacesPattern[-1] - xCentroids[-1])
    dBottomBoundary = 2.0 * (yFacesPattern[-1] - yCentroids[-1])

    # Assemble the distance vectors
    dLeftPattern = np.hstack([dLeftBoundary, xFacesPattern[1:] -
                              xFacesPattern[0:-1]])
    dRightPattern = np.hstack([xFacesPattern[1:] - xFacesPattern[0:-1],
                               dRightBoundary])
    dBottomPattern = np.hstack([yFacesPattern[1:] - yFacesPattern[0:-1],
                                dBottomBoundary])
    dTopPattern = np.hstack([dTopBoundary, yFacesPattern[1:] -
                             yFacesPattern[0:-1]])

    dLeft = np.tile(dLeftPattern[0:-1], numCellsWidth)
    dRight = np.tile(dRightPattern[1:], numCellsWidth)
    dBottom = np.repeat(dBottomPattern[1:], numCellsLength)
    dTop = np.repeat(dTopPattern[0:-1], numCellsLength)

    # Calculate the cell volumes. They are all have the same volume.
    cellLength = plateLength / numCellsLength
    cellWidth = plateWidth / numCellsWidth
    cellVolume = cellLength * cellWidth * thickness

    # Calculate the cross sectional area in the x and y directions
    areaX = cellWidth * thickness
    areaY = cellLength * thickness

    # Identify the cells which have boundary faces. Give them an ID of 1.
    topBoundaryID = np.hstack([np.ones(numCellsLength),
                               np.tile(np.zeros(numCellsLength),
                                       numCellsWidth - 1)])
    bottomBoundaryID = np.hstack([np.tile(np.zeros(numCellsLength),
                                          numCellsWidth - 1), np.ones(numCellsLength)])
    leftBoundaryID = np.tile(np.hstack([1.0, np.zeros(numCellsLength - 1)]),
                             numCellsWidth)
    rightBoundaryID = np.tile(np.hstack([np.zeros(numCellsLength - 1),
                                         1.0]), numCellsWidth)

    ###############################################################################
    #
    #     Calculate the Matrix Coefficients
    #
    ###############################################################################

    print(' Calculating Matrix Coefficients')
    print(lineSingle)

    # Diffusive flux per unit area
    DA_left = np.divide(conductivity * areaX, dLeft)
    DA_right = np.divide(conductivity * areaX, dRight)
    DA_bottom = np.divide(conductivity * areaY, dBottom)
    DA_top = np.divide(conductivity * areaY, dTop)

    # Source term Su
    # --------------
    # The volume heat flux is the same for interior and boundary cells
    Su = cellVolume * np.ones(numCells) * heatSourcePerVol

    # Add the contribution from each of the boundary faces
    Su += (2.0 * temperatureLeft * np.multiply(leftBoundaryID, DA_left))
    Su += (2.0 * temperatureRight * np.multiply(rightBoundaryID, DA_right))
    Su += (2.0 * temperatureBottom * np.multiply(bottomBoundaryID, DA_bottom))
    Su += (2.0 * temperatureTop * np.multiply(topBoundaryID, DA_top))

    # Source term Sp
    # ---------------
    # The source term is zero for interior cells
    Sp = np.zeros(numCells)

    # Add the contribution from each of the boundary faces
    Sp += (-2.0 * DA_left * leftBoundaryID)
    Sp += (-2.0 * DA_right * rightBoundaryID)
    Sp += (-2.0 * DA_bottom * bottomBoundaryID)
    Sp += (-2.0 * DA_top * topBoundaryID)

    # aL, aR, aT, aB
    # --------------

    # Only add contributions for interior cells
    aL = np.multiply(DA_left, 1 - leftBoundaryID)
    aR = np.multiply(DA_right, 1 - rightBoundaryID)
    aB = np.multiply(DA_bottom, 1 - bottomBoundaryID)
    aT = np.multiply(DA_top, 1 - topBoundaryID)

    # Calculate ap from the other coefficients
    aP = aL + aR + aB + aT - Sp

    ###############################################################################
    #
    #     Print the setup
    #
    ###############################################################################

    if (printSetup == 'true'):

        print(' Summary: Set Up')
        print(lineSingle)
        print('Cell | aL | aR  |  aB |  aT | Sp  |   Su  |  aP ')
        print(lineSingle)
        for i in range(numCells):
            print(('%4i %5.1f %5.1f %5.1f %5.1f %5.1f %7.1f %5.1f' % (
                i + 1, aL[i], aR[i], aB[i], aT[i], Sp[i], Su[i], aP[i])))
        print(lineSingle)

    ###############################################################################
    #
    #     Create the matrices
    #
    ###############################################################################

    print(' Assembling Matrices')
    print(lineSingle)

    # Start by creating an empty A matrix and an empty B matrix
    Amatrix = np.zeros([numCells, numCells])
    BVector = np.zeros(numCells)

    # Populate the matrix one row at a time (i.e one cell at a time)

    #
    # NOTE: this method is deliberately inefficient for this problem
    #        but it is useful for learning purposes. We could populate the
    #        diagonals and the off-diagonals directly.

    for i in range(numCells):
        BVector[i] = Su[i]

    for i in range(numCells):

        # Do the A matrix diagonal coefficients
        Amatrix[i, i] = aP[i]

        # Do the B matrix coefficients
        BVector[i] = Su[i]

        # Does the cell have a left boundary?
        if (leftBoundaryID[i] == 0.0):
            Amatrix[i, i - 1] = -1.0 * aL[i]

        # Does the cell have a right boundary?
        if (rightBoundaryID[i] == 0.0):
            Amatrix[i, i + 1] = -1.0 * aR[i]

        # Does the cell have a bottom boundary?
        if (bottomBoundaryID[i] == 0.0):
            Amatrix[i, i + numCellsLength] = -1.0 * aB[i]

        # Does the cell have a top boundary?
        if (topBoundaryID[i] == 0.0):
            Amatrix[i, i - numCellsLength] = -1.0 * aT[i]
    # --------------------------UNCOMMENT LAYER
    # ###############################################################################
    #
    #     Print the A matrix as a check
    #
    ###############################################################################

    # if (printSetup == 'true'):
    #     np.set_printoptions(linewidth=np.inf)
    #     print(Amatrix)
    #     print(lineSingle)

    ###############################################################################
    #
    #     Solve the matrices
    #
    ###############################################################################

    print(' Solving ...')
    print(lineSingle)

    return BVector, Nx, Ny, Amatrix
    # return BVector, cellLength, cellWidth, Amatrix
    # Use the built-in python solution module
    # Tvector = np.linalg.solve(Amatrix, BVector)
    #
    # # Reshape the vector into a grid
    # Tgrid = Tvector.reshape(numCellsWidth, numCellsLength)
    #
    # print(' Equations Solved')
    # print(lineSingle)
    #
    # ###############################################################################
    # #
    # #     Print the Results
    # #
    # ###############################################################################
    #
    # # Tap B Vector here.
    # # if (printSolution == 'true'):
    # #     print(' Solution: Temperature Field')
    # #     print(lineSingle)
    # #     print(rf"Temp without boundary:{Tgrid}")
    # #     print(lineSingle)
    #
    # ###############################################################################
    # #
    # #     Heat Fluxes
    # #
    # ###############################################################################
    #
    # # Calculate the temperature differences
    # # - To do this we need to stack on the boundary temperatures onto the grid
    # Tleftrightshift = np.hstack([temperatureLeft * np.ones([numCellsWidth, 1]),
    #                              Tgrid,
    #                              temperatureRight * np.ones([numCellsWidth, 1])])
    # Ttopbottomshift = np.vstack([temperatureTop * np.ones([numCellsLength]),
    #                              Tgrid,
    #                              temperatureBottom * np.ones([numCellsLength])])
    #
    # # Now we can calculate the temperature differences
    # deltaTleft = Tleftrightshift[:, 1:-1] - Tleftrightshift[:, 0:-2]
    # deltaTright = Tleftrightshift[:, 2:] - Tleftrightshift[:, 1:-1]
    # deltaTtop = Ttopbottomshift[0:-2, :] - Ttopbottomshift[1:-1, :]
    # deltaTbottom = Ttopbottomshift[1:-1, :] - Ttopbottomshift[2:, :]
    #
    # # We now need to calculate the diffusive heat flux (DA) on each face
    # # - Start by reshaping the DA vectors into a grid of the correct size
    # DA_left_grid = DA_left.reshape(numCellsWidth, numCellsLength)
    # DA_right_grid = DA_right.reshape(numCellsWidth, numCellsLength)
    # DA_top_grid = DA_top.reshape(numCellsWidth, numCellsLength)
    # DA_bottom_grid = DA_bottom.reshape(numCellsWidth, numCellsLength)
    #
    # # Calculate the boundary face fluxes
    # DA_left_boundary = ((2.0 * conductivity * areaX / dLeftBoundary) *
    #                     np.ones([numCellsWidth, 1]))
    # DA_right_boundary = ((2.0 * conductivity * areaX / dRightBoundary) *
    #                      np.ones([numCellsWidth, 1]))
    # DA_top_boundary = ((2.0 * conductivity * areaY / dTopBoundary) *
    #                    np.ones([numCellsLength]))
    # DA_bottom_boundary = ((2.0 * conductivity * areaY / dBottomBoundary) *
    #                       np.ones([numCellsLength]))
    #
    # # Now stack on the boundary face fluxes to the grid
    # DA_left_shift = np.hstack([DA_left_boundary, DA_left_grid[:, 1:]])
    # DA_right_shift = np.hstack([DA_right_grid[:, 0:-1], DA_right_boundary])
    # DA_top_shift = np.vstack([DA_top_boundary, DA_top_grid[1:, :]])
    # DA_bottom_shift = np.vstack([DA_bottom_grid[0:-1, :], DA_top_boundary])
    #
    # # Unit normal vectors
    # normalsLeftGrid = -1.0 * np.ones([numCellsWidth, numCellsLength])
    # normalsRightGrid = np.ones([numCellsWidth, numCellsLength])
    # normalsBottomGrid = -1.0 * np.ones([numCellsWidth, numCellsLength])
    # normalsTopGrid = np.ones([numCellsWidth, numCellsLength])
    #
    # # Now we can compute the heat fluxes
    # heatFluxLeft = -np.multiply(np.multiply(DA_left_shift, deltaTleft),
    #                             normalsLeftGrid)
    # heatFluxRight = -np.multiply(np.multiply(DA_right_shift, deltaTright),
    #                              normalsRightGrid)
    # heatFluxTop = -np.multiply(np.multiply(DA_top_shift, deltaTtop),
    #                            normalsTopGrid)
    # heatFluxBottom = -np.multiply(np.multiply(DA_bottom_shift, deltaTbottom),
    #                               normalsBottomGrid)
    #
    # # Calculate the volumetric heat source in each cell
    # sourceVol = heatSourcePerVol * cellVolume * np.ones([numCellsWidth,
    #                                                      numCellsLength])
    #
    # # Calculate the error in the heat flux balance in each cell
    # error = (sourceVol - heatFluxLeft - heatFluxRight - heatFluxTop -
    #          heatFluxBottom)
    #
    # # Reshape the matrices into vectors for printing
    # heatFluxLeftVector = heatFluxLeft.flatten()
    # heatFluxRightVector = heatFluxRight.flatten()
    # heatFluxTopVector = heatFluxTop.flatten()
    # heatFluxBottomVector = heatFluxBottom.flatten()
    # sourceVolVector = sourceVol.flatten()
    # errorVector = error.flatten()
    #
    # # if (printSolution == 'true'):
    # #
    # #     print(' Cell Heat Flux Balance')
    # #     print(lineSingle)
    # #     print(' Cell|  QL   |  QR   |  QT   |  QB   |  SV   |  Err')
    # #     print(lineSingle)
    # #     for i in range(numCells):
    # #         print(('%4i %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f' % (
    # #             i + 1, heatFluxLeftVector[i], heatFluxRightVector[i],
    # #             heatFluxTopVector[i], heatFluxBottomVector[i],
    # #             sourceVolVector[i], errorVector[i])))
    # #     print(lineSingle)
    #
    # # Sum the heat fluxes across the boundary faces to give the total heat flux
    # # across each boundary
    # heatFluxLeftBoundaryTotal = np.sum(np.multiply(leftBoundaryID,
    #                                                heatFluxLeftVector))
    # heatFluxRightBoundaryTotal = np.sum(np.multiply(rightBoundaryID,
    #                                                 heatFluxRightVector))
    # heatFluxBottomBoundaryTotal = np.sum(np.multiply(bottomBoundaryID,
    #                                                  heatFluxBottomVector))
    # heatFluxTopBoundaryTotal = np.sum(np.multiply(topBoundaryID,
    #                                               heatFluxTopVector))
    # heatFluxBoundaryTotal = (heatFluxLeftBoundaryTotal
    #                          + heatFluxRightBoundaryTotal
    #                          + heatFluxTopBoundaryTotal
    #                          + heatFluxBottomBoundaryTotal)
    # heatGeneratedTotal = np.sum(sourceVolVector)
    #
    # if (printSolution == 'true'):
    #     print(' Boundary Heat Flux Balance')
    #     print(lineSingle)
    #     print((' Left      : %7.1f [W]' % (heatFluxLeftBoundaryTotal)))
    #     print((' Right     : %7.1f [W]' % (heatFluxRightBoundaryTotal)))
    #     print((' Bottom    : %7.1f [W]' % (heatFluxBottomBoundaryTotal)))
    #     print((' Top       : %7.1f [W]' % (heatFluxTopBoundaryTotal)))
    #     print((' Total     : %7.1f [W]' % (heatFluxBoundaryTotal)))
    #     print(lineSingle)
    #     print((' Generated : %7.1f [W]' % (heatGeneratedTotal)))
    #     print(lineSingle)
    #     print((' Error     : %7.1f [W]' % (heatFluxBoundaryTotal -
    #                                        heatGeneratedTotal)))
    #     print(lineSingle)
    #
    # ###############################################################################
    # #
    # #     Interpolate the node temperatures
    # #
    # ###############################################################################
    #
    # # Interpolate the solution on the interior nodes from the CFD solution
    # Tleftrightnodes = 0.5 * (Tgrid[:, 1:] + Tgrid[:, :-1])
    # Tinternalnodes = 0.5 * (Tleftrightnodes[1:, :] + Tleftrightnodes[:-1, :])
    #
    # # Interpolate the boundary temperatures in the corners
    # temperatureTopLeftCorner = 0.5 * (temperatureTop + temperatureLeft)
    # temperatureTopRightCorner = 0.5 * (temperatureTop + temperatureRight)
    # temperatureBottomLeftCorner = 0.5 * (temperatureBottom + temperatureLeft)
    # temperatureBottomRightCorner = 0.5 * (temperatureBottom + temperatureRight)
    #
    # # Assemble the temperatures on all the boundary nodes
    # temperatureTopVector = np.hstack([temperatureTopLeftCorner,
    #                                   temperatureTop * np.ones(numCellsLength - 1),
    #                                   temperatureTopRightCorner])
    # temperatureBottomVector = np.hstack([temperatureBottomLeftCorner,
    #                                      temperatureBottom *
    #                                      np.ones(numCellsLength - 1),
    #                                      temperatureBottomRightCorner])
    # temperatureLeftVector = temperatureLeft * np.ones([numCellsWidth - 1, 1])
    # temperatureRightVector = temperatureRight * np.ones([numCellsWidth - 1, 1])
    #
    # # Assemble the temperature on all of the nodes together as one grid
    # Tnodes = np.vstack([temperatureTopVector, np.hstack([temperatureLeftVector,
    #                                                      Tinternalnodes, temperatureRightVector]),
    #                     temperatureBottomVector])
    #
    # # X and Y coordinates of the nodes
    # xNodes = xFaces.reshape([numCellsWidth + 1, numCellsLength + 1])
    # yNodes = np.flipud(yFaces.reshape([numCellsWidth + 1, numCellsLength + 1]))
    #
    # ###############################################################################
    # #
    # #     Plot the solution
    # #
    # ###############################################################################
    #
    # # Plot the data if desired
    # if (plotOutput == 'true'):
    #     print(' Plotting ...')
    #     print(lineSingle)
    #
    #     # plt.imshow(Tnodes, vmax=Tnodes.min(), vmin=Tnodes.max(), cmap='RdBu', origin='lower')
    #     plt.imshow(Tnodes, vmax=Tnodes.min(), vmin=Tnodes.max(), cmap='RdBu', origin='lower')
    #     plt.colorbar()
    #     plt.title(rf"Gridsize: {numCellsLength}*{numCellsWidth}_convential_method")
    #     plt.savefig('convential_condution_soln'+ '.png', dpi=300)
    #     plt.close()
    # return BVector, cellLength, cellWidth, Tnodes
    # #


# a,b,c,d = diffusion_eqn()