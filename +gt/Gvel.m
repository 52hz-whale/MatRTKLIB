classdef Gvel < handle
    % Gvel: GNSS velocity class
    %
    % Gvel Declaration:
    % obj = Gvel(vel, 'type', [orgpos], ['orgtype'])
    %   vel      : Nx3, velocity vector
    %                [ECEF x(m/s), ECEF y(m/s), ECEF z(m/s)] or
    %                [east(m/s), north(m/s), up(m/s)]
    %   veltype  : Coordinate type: 'xyz' or 'enu'
    %   [orgpos] : 1x3, Coordinate origin position vector
    %                [latitude(deg), longitude(deg), ellipsoidal height(m)] or
    %                [ECEF x(m), ECEF y(m), ECEF z(m)]
    %   [orgtype]: 1x1, Coordinate type: 'llh' or 'xyz'
    %
    % Gvel Properties:
    %   n       : 1x1, Number of epochs
    %   xyz     : (obj.n)x3, ECEF velocity (m/s, m/s, m/s)
    %   enu     : (obj.n)x3, Local ENU velocity (m/s, m/s, m/s)
    %   orgllh  : 1x3, Coordinate origin (deg, deg, m)
    %   orgxyz  : 1x3, Coordinate origin in ECEF (m, m, m)
    %   v2      : (obj.n)x1, Horizontal (2D) velocity (m/s)
    %   v3      : (obj.n)x1, 3D velocity (m/s)
    %
    % Gvel Methods:
    %   setVel(vel, veltype): set velocity
    %   setOrg(pos, postype): set coordinate orgin
    %   append(gvel): Append GNSS velocity
    %   addOffset(offset, [coordtype]): Add offset to the velocity data
    %   gerr = difference(gvel): Compute the difference between two Gvel objects
    %   gpos = integral(dt, [idx]): cumulative integral
    %   gvel = select([idx]): Select from index 
    %   [mxyz, sdxyz] = meanXYZ([idx]): Compute the mean and standard deviation of XYZ velocity
    %   [menu, sdenu] = meanENU([idx]): Compute the mean and standard deviation of ENU velocity
    %   [m2d, sd2d] = mean2D([idx]): Compute the mean and standard deviation of 2D velocity 
    %   [m3d, sd3d] = mean3D([idx]): Compute the mean and standard deviation of 3D velocity
    %   x = x([idx]): Get the x-component of the velocity data
    %   y = y([idx]): Get the y-component of the velocity data
    %   z = z([idx]): Get the z-component of the velocity data
    %   east = east([idx]): Get the east-component of the ENU velocity data
    %   north = north([idx]): Get the north-component of the ENU velocity data
    %   up = up([idx]): Get the up-component of the ENU velocity data
    %   plot([idx]): Plot ENU velocity
    %   plotXYZ([idx]): Plot XYZ velocity
    %   plot2D([idx]): Plot 2D velocity
    %   plot3D([idx]): Plot 3D velocity 
    %   help()
    %
    % Gpos Overloads:
    %   gerr = obj - gvel
    %
    % Author: Taro Suzuki

    properties
        n % Number of epochs
        xyz % ECEF velocity (m/s, m/s, m/s)
        enu % Local ENU velocity (m/s, m/s, m/s)
        orgllh % Coordinate origin (deg, deg, m)
        orgxyz % Coordinate origin in ECEF (m, m, m)
        v2 % Horizontal (2D) velocity (m/s)
        v3 % 3D velocity (m/s)
    end

    methods
        %% constractor
        function obj = Gvel(vel, veltype, org, orgtype)
            arguments
                vel (:,3) double
                veltype (1,:) char {mustBeMember(veltype,{'xyz','enu'})}
                org (1,3) double = [0, 0, 0]
                orgtype (1,:) char {mustBeMember(orgtype,{'llh','xyz'})} = 'llh'
            end
            if nargin>=2; obj.setVel(vel, veltype); end
            if nargin==4; obj.setOrg(org, orgtype); end
        end

        %% set velocity
        function setVel(obj, vel, veltype)
            % setVel : set velocity
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.plot(org, vel, veltype)
            %
            % Input: ------------------------------------------------------
            %  vel : velocity vector
            %  veltype : Coordinate type: 'xyz' or 'enu'
            % 
            arguments
                obj gt.Gvel
                vel (:,3) double
                veltype (1,:) char {mustBeMember(veltype,{'xyz','enu'})}
            end
            obj.n = size(vel,1);
            switch veltype
                case 'xyz'
                    obj.xyz = vel;
                    if ~isempty(obj.orgllh); obj.enu = rtklib.ecef2enu(obj.xyz, obj.orgllh); end
                case 'enu'
                    obj.enu = vel;
                    if ~isempty(obj.orgllh); obj.xyz = rtklib.enu2ecef(obj.enu, obj.orgllh); end
            end
            if ~isempty(obj.enu)
                obj.v2 = vecnorm(obj.enu(:,1:2), 2, 2);
                obj.v3 = vecnorm(obj.enu, 2, 2);
            else
                obj.v3 = vecnorm(obj.xyz, 2, 2);
            end
        end

        %% set coordinate orgin
        function setOrg(obj, org, orgtype)
            % setOrg : set coordinate orgin
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.setOrg(org, orgtype)
            %
            % Input: ------------------------------------------------------
            %  org : Coordinate origin
            %  orgtype : Coordinate type: 'llh' or 'xyz'
            % 
            arguments
                obj gt.Gvel
                org (1,3) double
                orgtype (1,:) char {mustBeMember(orgtype,{'llh','xyz'})}
            end
            switch orgtype
                case 'llh'
                    obj.orgllh = org;
                    obj.orgxyz = rtklib.llh2xyz(org);
                case 'xyz'
                    obj.orgxyz = org;
                    obj.orgllh = rtklib.xyz2llh(org);
            end
            if ~isempty(obj.xyz)
                obj.enu = rtklib.ecef2enu(obj.xyz, obj.orgllh);
            elseif ~isempty(obj.enu)
                obj.xyz = rtklib.enu2ecef(obj.enu, obj.orgllh);
            end
            obj.v2 = vecnorm(obj.enu(:,1:2), 2, 2);
            obj.v3 = vecnorm(obj.enu, 2, 2);
        end

        %% append
        function append(obj, gvel)
            % append : Append GNSS velocity
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.append(gvel)
            %
            % Input: ------------------------------------------------------
            %  gvel : GNSS velocity class
            % 
            arguments
                obj gt.Gvel
                gvel gt.Gvel
            end
            if ~isempty(obj.xyz)
                obj.setVel([obj.xyz; gvel.xyz], 'xyz');
            else
                obj.setVel([obj.enu; gvel.enu], 'enu');
            end
        end

        %% addOffset
        function addOffset(obj, offset, coordtype)
            % addOffset: Add offset to the velocity data
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.addOffset(offset, coordtype)
            %
            % Input: ------------------------------------------------------
            %  offset : Velocity offset vector
            %  coordtype : Coordinate type: 'xyz' or 'enu'
            % 
            arguments
                obj gt.Gvel
                offset (1,3) double
                coordtype (1,:) char {mustBeMember(coordtype,{'enu','xyz'})} = 'enu'
            end
            switch coordtype
                case 'enu'
                    if isempty(obj.enu)
                        error('enu must be set to a value');
                    end
                    obj.setVel(obj.enu + offset, 'enu');
                case 'xyz'
                    if isempty(obj.xyz)
                        error('xyz must be set to a value');
                    end
                    obj.setVel(obj.xyz + offset, 'xyz');
            end
        end

        %% difference
        function gerr = difference(obj, gvel)
            % difference : Compute the difference between two Gvel objects
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.difference(gvel)
            %
            % Input: ------------------------------------------------------
            %  gvel :  GNSS velocity class
            % 
            arguments
                obj gt.Gvel
                gvel gt.Gvel
            end
            if obj.n ~= gvel.n
                error('size of the two gt.Gvel must be same')
            end
            if ~isempty(obj.xyz) && ~isempty(gvel.xyz)
                gerr = gt.Gerr('velocity', obj.xyz - gvel.xyz, 'xyz');
                if ~isempty(obj.orgllh); gerr.setOrg(obj.orgllh, 'llh'); end
            elseif ~isempty(obj.enu) && ~isempty(gvel.enu)
                gerr = gt.Gerr('velocity', obj.enu - gvel.enu, 'enu');
                if ~isempty(obj.orgllh); gerr.setOrg(obj.orgllh, 'llh'); end
            else
                error('two gt.Gvel must have both xyz or enu')
            end
        end

        %% cumulative integral
        function gpos = integral(obj, dt, idx)
            % integral: cumulative integral
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.integral(dt, idx)
            %
            % Input: ------------------------------------------------------
            %  dt : Time step
            %  idx : Logical or numeric index to select
            % 
            % Output: ------------------------------------------------------
            %  gpos :  GNSS position class
            % 
            arguments
                obj gt.Gvel
                dt (1,1) double
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if ~isempty(obj.xyz)
                xyz_ = cumtrapz(dt*obj.xyz(idx,:));
                gpos = gt.Gpos(xyz_, 'xyz');
            else
                enu_ = cumtrapz(dt*obj.enu(idx,:));
                gpos = gt.Gpos(enu_, 'enu');
            end
            if ~isempty(obj.orgllh); gvel.setOrg(obj.orgllh, 'llh'); end
        end
        
        %% copy
        function gvel = copy(obj)
            % copy: Copy object
            % -------------------------------------------------------------
            % MATLAB handle class is used, so if you want to create a
            % different instance, you need to use the copy method.
            %
            % Usage: ------------------------------------------------------
            %   gtime = obj.copy()
            %
            % Output: ------------------------------------------------------
            %   gtime : 1x1, Copied object
            %
            arguments
                obj gt.Gvel
            end
            gvel = obj.select(1:obj.n);
        end

        %% select from index
        function gvel = select(obj, idx)
            % select : Select from index 
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.select([idx])
            %
            % Input: ------------------------------------------------------
            %  [idx] : Logical or numeric index to select
            % 
            % Output: ------------------------------------------------------
            %  gvel :  GNSS velocity class
            % 
            arguments
                obj gt.Gvel
                idx {mustBeInteger, mustBeVector}
            end
            if ~any(idx)
                error('Selected index is empty');
            end
            if ~isempty(obj.xyz)
                gvel = gt.Gvel(obj.xyz(idx,:), 'xyz');
            else
                gvel = gt.Gvel(obj.enu(idx,:), 'enu');
            end
            if ~isempty(obj.orgllh); gvel.setOrg(obj.orgllh, 'llh'); end
        end

        %% mean calculation
        function [mxyz, sdxyz] = meanXYZ(obj, idx)
            % select : Compute the mean and standard deviation of XYZ velocity
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.meanXYZ([idx])
            %
            % Input: ------------------------------------------------------
            %  [idx] : Logical or numeric index to select
            % 
            % Output: ------------------------------------------------------
            %  mxyz :  Mean of ECEF velocities
            %  sdxyz : Standard deviation of ECEF velocities
            % 
            arguments
                obj gt.Gvel
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.xyz)
                error('xyz must be set to a value');
            end
            mxyz = mean(obj.xyz(idx,:), 1, 'omitnan');
            sdxyz = std(obj.xyz(idx,:), 0, 1, 'omitnan');
        end
        function [menu, sdenu] = meanENU(obj, idx)
            % select :  Compute the mean and standard deviation of ENU velocity
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.meanENU([idx])
            %
            % Input: ------------------------------------------------------
            %  [idx] : Logical or numeric index to select
            % 
            % Output: ------------------------------------------------------
            %  menu :  Mean of ENU velocities
            %  sdenu : Standard deviation of ENU velocity
            % 
            arguments
                obj gt.Gvel
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.enu)
                error('enu must be set to a value');
            end
            menu = mean(obj.enu(idx,:), 1, 'omitnan');
            sdenu = std(obj.enu(idx,:), 0, 1, 'omitnan');
        end
        function [m2d, sd2d] = mean2D(obj, idx)
            % mean2D : Compute the mean and standard deviation of 2D velocity 
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.mean2D([idx])
            %
            % Input: ------------------------------------------------------
            %  [idx] : Logical or numeric index to select 
            % 
            % Output: ------------------------------------------------------
            %  m2d :  Mean of horizontal velocities
            %  sd2d : Standard deviation of horizontal velocity
            % 
            arguments
                obj gt.Gvel
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.v2)
                error('enu must be set to a value');
            end
            m2d = mean(obj.v2(idx), 1, 'omitnan');
            sd2d = std(obj.v2(idx), 0, 1, 'omitnan');
        end
        function [m3d, sd3d] = mean3D(obj, idx)
            % mean3D : Compute the mean and standard deviation of 3D velocity
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.mean3D([idx])
            %
            % Input: ------------------------------------------------------
            %  [idx] : Logical or numeric index to select
            % 
            % Output: ------------------------------------------------------
            %  m3d :  Mean of 3D velocities
            %  sd3d :  Standard deviation of 3D velocity
            % 
            arguments
                obj gt.Gvel
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            m3d = mean(obj.v3(idx), 1, 'omitnan');
            sd3d = std(obj.v3(idx), 0, 1, 'omitnan');
        end

        %% access
        function x = x(obj, idx)
            % x : Get the x-component of the velocity data
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.x([idx])
            %
            % Input: ------------------------------------------------------
            %  [idx] : Logical or numeric index to select
            % 
            % Output: ------------------------------------------------------
            %  x :  x-component of the velocity data
            % 
            arguments
                obj gt.Gvel
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.xyz)
                error('xyz must be set to a value');
            end
            x = obj.xyz(idx,1);
        end
        function y = y(obj, idx)
            % y : Get the y-component of the velocity data
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.y([idx])
            %
            % Input: ------------------------------------------------------
            %  [idx] : Logical or numeric index to select
            % 
            % Output: ------------------------------------------------------
            %  y :  y-component of the velocity data
            % 
            arguments
                obj gt.Gvel
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.xyz)
                error('xyz must be set to a value');
            end
            y = obj.xyz(idx,2);
        end
        function z = z(obj, idx)
            % z : Get the z-component of the velocity data
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.z([idx])
            %
            % Input: ------------------------------------------------------
            %  [idx] : Logical or numeric index to select
            % 
            % Output: ------------------------------------------------------
            %  z :  z-component of the velocity data
            %
            arguments
                obj gt.Gvel
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.xyz)
                error('xyz must be set to a value');
            end
            z = obj.xyz(idx,3);
        end
        function east = east(obj, idx)
            % east : Get the east-component of the ENU velocity data
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.east([idx])
            %
            % Input: ------------------------------------------------------
            %  [idx] : Logical or numeric index to select
            % 
            % Output: ------------------------------------------------------
            %  east :  east-component of the ENU velocity data
            %
            arguments
                obj gt.Gvel
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.enu)
                error('enu must be set to a value');
            end
            east = obj.enu(idx,1);
        end
        function north = north(obj, idx)
            % north : Get the north-component of the ENU velocity data
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.north([idx])
            %
            % Input: ------------------------------------------------------
            %  [idx] : Logical or numeric index to select
            % 
            % Output: ------------------------------------------------------
            %  north :  north-component of the ENU velocity data
            %
            arguments
                obj gt.Gvel
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.enu)
                error('enu must be set to a value');
            end
            north = obj.enu(idx,2);
        end
        function up = up(obj, idx)
            % up : Get the up-component of the ENU velocity data
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.up([idx])
            %
            % Input: ------------------------------------------------------
            %  [idx] : Logical or numeric index to select
            % 
            % Output: ------------------------------------------------------
            %  up :  up-component of the ENU velocity data
            %
            arguments
                obj gt.Gvel
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.enu)
                error('enu must be set to a value');
            end
            up = obj.enu(idx,3);
        end

        %% plot
        function plot(obj, idx)
            % plot : Plot ENU velocity
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.plot([idx])
            %
            % Input: ------------------------------------------------------
            %  [idx] : Logical or numeric index to select
            % 
            arguments
                obj gt.Gvel
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.enu)
                error('enu must be set to a value');
            end
            figure;
            tiledlayout(3,1,'TileSpacing','Compact');
            nexttile;
            plot(obj.enu(idx, 1), '.-');
            ylabel('East (m/s)');
            grid on;
            nexttile;
            plot(obj.enu(idx, 2), '.-');
            ylabel('North (m/s)');
            grid on;
            nexttile;
            plot(obj.enu(idx, 3), '.-');
            ylabel('Up (m/s)');
            grid on;
        end
        function plotXYZ(obj, idx)
            % plotXYZ : Plot XYZ velocity
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.plotXYZ([idx])
            %
            % Input: ------------------------------------------------------
            %  [idx] : Logical or numeric index to select
            % 
            arguments
                obj gt.Gvel
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.xyz)
                error('enu must be set to a value');
            end
            figure;
            tiledlayout(3,1,'TileSpacing','Compact');
            a1 = nexttile;
            plot(obj.xyz(idx, 1), '.-');
            ylabel('X (m/s)');
            grid on;
            a2 = nexttile;
            plot(obj.xyz(idx, 2), '.-');
            ylabel('Y (m/s)');
            grid on;
            a3 = nexttile;
            plot(obj.xyz(idx, 3), '.-');
            ylabel('Z (m/s)');
            grid on;

            linkaxes([a1 a2 a3],'x');
            drawnow
        end
        
        function plot2D(obj, idx)
            % plot2D : Plot 2D velocity
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.plot2D([idx])
            %
            % Input: ------------------------------------------------------
            %  [idx] : Logical or numeric index to select
            % 
            arguments
                obj gt.Gvel
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.v2)
                error('enu must be set to a value');
            end
            figure;
            plot(obj.v2(idx), '.-');
            ylabel('Horizontal velocity (m/s)');
            grid on;
            drawnow
        end

        function plot3D(obj, idx)
            % plot3D : Plot 3D velocity 
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.plot3D([idx])
            %
            % Input: ------------------------------------------------------
            %  [idx] : Logical or numeric index to select
            % 
            arguments
                obj gt.Gvel
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            figure;
            plot(obj.v3(idx), '.-');
            ylabel('3D velocity (m/s)');
            grid on;
            drawnow
        end

        %% help
        function help(~)
            doc gt.Gvel
        end

        %% overload
        function gerr = minus(obj, gvel)
            % minus: Subtract two Gvel objects
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.minus([gvel])
            %
            % Input: ------------------------------------------------------
            %  gvel : GNSS velocity class
            % 
            % Output: -----------------------------------------------------
            %  gerr : Resulting Gvel object after subtraction 
            % 
            gerr = obj.difference(gvel);
        end
    end
end