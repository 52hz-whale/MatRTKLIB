classdef Gpos < handle
    % Gpos: GNSS position class
    %
    % Gpos Declaration:
    % obj = Gpos(pos, 'type', [orgpos], ['orgtype'])
    %   pos      : Nx3, postion vector
    %                [latitude(deg), longitude(deg), ellipsoidal height(m)] or
    %                [ECEF x(m), ECEF y(m), ECEF z(m)] or
    %                [east(m), north(m), up(m)]
    %   type     : 1x1, Coordinate type: 'llh' or 'xyz' or 'enu'
    %   [orgpos] : 1x3, Coordinate origin 1x3 position vector
    %   [orgtype]: 1x1, Coordinate type: 'llh' or 'xyz'
    %
    % Gpos Properties:
    %   n       : 1x1, Number of epochs
    %   llh     : (obj.n)x3, Geodetic position (deg, deg, m)
    %   xyz     : (obj.n)x3, ECEF position (m, m, m)
    %   enu     : (obj.n)x3, Local ENU position (m, m, m)
    %   orgllh  : 1x3, Coordinate origin (deg, deg, m)
    %   orgxyz  : 1x3, Coordinate origin in ECEF (m, m, m)
    %
    % Gpos Methods:
    %   setPos(pos, type):
    %   setOrg(pos, type):
    %   append(gpos):
    %   addOffset(offset, [coordtype]):
    %   gerr = difference(gpos):
    %   gvel = gradient(dt, [idx]):
    %   gpos = select(idx):
    %   outpos(file, type, idx):
    %   [gpos, gcov] = mean([idx]);
    %   [mllh, sdenu] = meanLLH([idx]):
    %   [mxyz, sdxyz] = meanXYZ([idx]):
    %   [menu, sdenu] = meanENU([idx]):
    %   llhrad = llhRad([idx]):
    %   latrad = latRad([idx]):
    %   lonrad = lonRad([idx]):
    %   llhdms = llhDMS([idx]):
    %   latdms = latDMS([idx]):
    %   londms = lonDMS([idx]):
    %   gh = geoid([idx]):
    %   oh = orthometric([idx]):
    %   lat = lat([idx]):
    %   lon = lon([idx]):
    %   h = h([idx]):
    %   x = x([idx]):
    %   y = y([idx]):
    %   z = z([idx]):
    %   east = east([idx]):
    %   north = north([idx]):
    %   up = up([idx]):
    %   plot([idx]):
    %   help()
    %
    % Gpos Overloads:
    %   gerr = obj - gpos
    %
    % Author: Taro Suzuki

    properties
        n, llh, xyz, enu, orgllh, orgxyz;
    end

    methods
        %% constractor
        function obj = Gpos(pos, postype, org, orgtype)
            arguments
                pos (:,3) double
                postype (1,:) char {mustBeMember(postype,{'llh','xyz','enu'})}
                org (1,3) double = [0, 0, 0]
                orgtype (1,:) char {mustBeMember(orgtype,{'llh','xyz'})} = 'llh'
            end
            if nargin>=2; obj.setPos(pos, postype); end
            if nargin==4; obj.setOrg(org, orgtype); end
        end

        %% set position
        function setPos(obj, pos, postype)
            arguments
                obj gt.Gpos
                pos (:,3) double
                postype (1,:) char {mustBeMember(postype,{'llh','xyz','enu'})}
            end
            obj.n = size(pos,1);
            switch postype
                case 'llh'
                    obj.llh = pos;
                    obj.xyz = rtklib.llh2xyz(obj.llh);
                    if ~isempty(obj.orgllh); obj.enu = rtklib.llh2enu(obj.llh, obj.orgllh); end
                case 'xyz'
                    obj.xyz = pos;
                    obj.llh = rtklib.xyz2llh(obj.xyz);
                    if ~isempty(obj.orgllh); obj.enu = rtklib.xyz2enu(obj.xyz, obj.orgllh); end
                case 'enu'
                    obj.enu = pos;
                    if ~isempty(obj.orgllh)
                        obj.llh = rtklib.enu2llh(obj.enu, obj.orgllh);
                        obj.xyz = rtklib.enu2xyz(obj.enu, obj.orgllh);
                    end
            end
        end

        %% set coordinate orgin
        function setOrg(obj, org, orgtype)
            arguments
                obj gt.Gpos
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
            if ~isempty(obj.llh)
                obj.enu = rtklib.llh2enu(obj.llh, obj.orgllh);
            elseif ~isempty(obj.xyz)
                obj.enu = rtklib.xyz2enu(obj.xyz, obj.orgllh);
            elseif ~isempty(obj.enu)
                obj.llh = rtklib.enu2llh(obj.enu, obj.orgllh);
                obj.xyz = rtklib.enu2xyz(obj.enu, obj.orgllh);
            end
        end

        %% insert
        function insert(obj, idx, gpos)
            arguments
                obj gt.Gpos
                idx (1,1) {mustBeInteger}
                gpos gt.Gpos
            end
            if idx<=0 || idx>obj.n
                error('Index is out of range');
            end
            if ~isempty(obj.llh) && ~isempty(gpos.llh)
                obj.setPos(obj.insertdata(obj.llh,idx,gpos.llh), 'llh');
            else
                obj.setPos(obj.insertdata(obj.enu,idx,gpos.enu), 'enu');
            end
        end

        %% append
        function append(obj, gpos)
            arguments
                obj gt.Gpos
                gpos gt.Gpos
            end
            if ~isempty(obj.llh) && ~isempty(gpos.llh)
                obj.setPos([obj.llh; gpos.llh], 'llh');
            else
                obj.setPos([obj.enu; gpos.enu], 'enu');
            end
        end

        %% addOffset
        function addOffset(obj, offset, coordtype)
            arguments
                obj gt.Gpos
                offset (1,3) double
                coordtype (1,:) char {mustBeMember(coordtype,{'enu','xyz'})} = 'enu'
            end
            switch coordtype
                case 'enu'
                    if ~isempty(obj.enu)
                        obj.setPos(obj.enu + offset, 'enu');
                    else
                        % first position is origin
                        enu_ = rtklib.llh2enu(obj.llh, obj.llh(1,:));
                        llh_ = rtklib.enu2llh(enu_ + offset, obj.llh(1,:));
                        obj.setPos(llh_, 'llh');
                    end
                case 'xyz'
                    if isempty(obj.xyz)
                        error('xyz must be set to a value');
                    end
                    obj.setPos(obj.xyz + offset, 'xyz');
            end
        end

        %% difference
        function gerr = difference(obj, gpos)
            arguments
                obj gt.Gpos
                gpos gt.Gpos
            end
            if obj.n ~= gpos.n
                error('size of the two gt.Gpos must be same')
            end
            if ~isempty(obj.xyz) && ~isempty(gpos.xyz)
                gerr = gt.Gerr('position', obj.xyz - gpos.xyz, 'xyz');
            elseif ~isempty(obj.enu) && ~isempty(gpos.enu)
                gerr = gt.Gerr('position', obj.enu - gpos.enu, 'enu');
            else
                error('two gt.Gpos must have both xyz or enu')
            end
            if ~isempty(gpos.orgllh)
                gerr.setOrg(gpos.orgllh, 'llh')
            elseif ~isempty(obj.orgllh)
                gerr.setOrg(obj.orgllh, 'llh')
            end
        end

        %% gradient
        function gvel = gradient(obj, dt, idx)
            arguments
                obj gt.Gpos
                dt (1,1) double
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if ~isempty(obj.xyz)
                vxyz = [gradient(obj.xyz(idx,1)),...
                    gradient(obj.xyz(idx,2)),...
                    gradient(obj.xyz(idx,3))]/dt;
                gvel = gt.Gvel(vxyz, 'xyz');
            else
                venu = [gradient(obj.enu(idx,1)),...
                    gradient(obj.enu(idx,2)),...
                    gradient(obj.enu(idx,3))]/dt;
                gvel = gt.Gvel(venu, 'enu');
            end
            if ~isempty(obj.orgllh); gvel.setOrg(obj.orgllh, 'llh'); end
        end
        
        %% copy
        function gpos = copy(obj)
            arguments
                obj gt.Gpos
            end
            gpos = obj.select(1:obj.n);
        end
        
        %% select
        function gpos = select(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector}
            end
            if ~any(idx)
                error('Selected index is empty');
            end
            if ~isempty(obj.llh)
                gpos = gt.Gpos(obj.llh(idx,:), 'llh');
            else
                gpos = gt.Gpos(obj.enu(idx,:), 'enu');
            end
            if ~isempty(obj.orgllh); gpos.setOrg(obj.orgllh, 'llh'); end
        end

        %% output position
        function outpos(obj, file, type, idx)
            arguments
                obj gt.Gpos
                file (1,:) char
                type (1,:) char {mustBeMember(type,{'llh','llhdms','xyz','enu'})}
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            gpos = obj.select(idx);
            llhdms = gpos.llhDMS();
            fid = fopen(file,'w');
            for i=1:gpos.n
                switch type
                    case 'llh'
                        fprintf(fid, '%.8f %.8f %.4f\n', gpos.llh(i,1), gpos.llh(i,2), gpos.llh(i,3));
                    case 'llhdms'
                        fprintf(fid, '%.0f %.0f  %.6f ', llhdms{1}(i,1), llhdms{1}(i,2), llhdms{1}(i,3));
                        fprintf(fid, '%.0f %.0f  %.6f ', llhdms{2}(i,1), llhdms{2}(i,2), llhdms{2}(i,3));
                        fprintf(fid, '%.4f\n', llhdms{3}(i));
                    case 'xyz'
                        fprintf(fid, '%.4f %.4f %.4f\n', gpos.xyz(i,1), gpos.xyz(i,2), gpos.xyz(i,3));
                    case 'enu'
                        fprintf(fid, '%.4f %.4f %.4f\n', gpos.enu(i,1), gpos.enu(i,2), gpos.enu(i,3));
                end
            end
            fclose(fid);
        end

        %% mean calculation
        function [gpos, gcov] = mean(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.llh)
                menu = mean(obj.enu(idx,:), 1, 'omitnan');
                gpos = gt.Gpos(menu, 'enu');
            else
                mllh = mean(obj.llh(idx,:), 1, 'omitnan');
                if isempty(obj.orgllh)
                    gpos = gt.Gpos(mllh, 'llh', mllh, 'llh');
                else
                    gpos = gt.Gpos(mllh, 'llh', obj.orgllh, 'llh');
                end
            end
            gcov = gt.Gcov(obj);            
        end
        function [mllh, sdenu] = meanLLH(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.llh)
                error('llh must be set to a value');
            end
            mllh = mean(obj.llh(idx,:), 1, 'omitnan');
            enu_ = rtklib.llh2enu(obj.llh, obj.llh(1,:));
            sdenu = std(enu_, 0, 1, 'omitnan');
        end
        function [mxyz, sdxyz] = meanXYZ(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.xyz)
                error('xyz must be set to a value');
            end
            mxyz = mean(obj.xyz(idx,:), 1, 'omitnan');
            sdxyz = std(obj.xyz(idx,:), 0, 1, 'omitnan');
        end
        function [menu, sdenu] = meanENU(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.enu)
                error('enu must be set to a value');
            end
            menu = mean(obj.enu(idx,:), 1, 'omitnan');
            sdenu = std(obj.enu(idx,:), 0, 1, 'omitnan');
        end

        %% to radian
        function llhrad = llhRad(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.llh)
                error('llh must be set to a value');
            end
            llhrad = [obj.llh(idx,1)/180*pi, obj.llh(idx,2)/180*pi, obj.llh(idx,3)];
        end
        function latrad = latRad(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.llh)
                error('llh must be set to a value');
            end
            latrad = obj.llh(idx,1)/180*pi;
        end
        function lonrad = lonRad(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.llh)
                error('llh must be set to a value');
            end
            lonrad = obj.llh(idx,2)/180*pi;
        end

        %% to dgree, minute, and second
        function llhdms = llhDMS(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.llh)
                error('llh must be set to a value');
            end
            latdms = rtklib.deg2dms(obj.lat(idx));
            londms = rtklib.deg2dms(obj.lon(idx));
            llhdms = {latdms, londms, obj.h(idx)};
        end
        function latdms = latDMS(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.llh)
                error('llh must be set to a value');
            end
            latdms = rtklib.deg2dms(obj.lat(idx));
        end
        function londms = lonDMS(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.llh)
                error('llh must be set to a value');
            end
            londms = rtklib.deg2dms(obj.lon(idx));
        end

        %% geoid function
        function gh = geoid(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.llh)
                error('llh must be set to a value');
            end
            gpos = obj.select(idx);
            gh = NaN(gpos.n, 1);
            idx = ~any(isnan(gpos.llh(:,1:2)),2);
            gh_ = rtklib.geoidh(gpos.llh(idx,1), gpos.llh(idx,2));
            gh(idx) = gh_;
        end
        function oh = orthometric(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.llh)
                error('llh must be set to a value');
            end
            oh = obj.llh(idx,3) - obj.geoid(idx);
        end

        %% access
        function lat = lat(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.llh)
                error('llh must be set to a value');
            end
            lat = obj.llh(idx,1);
        end
        function lon = lon(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.llh)
                error('llh must be set to a value');
            end
            lon = obj.llh(idx,2);
        end
        function h = h(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.llh)
                error('llh must be set to a value');
            end
            h = obj.llh(idx,3);
        end
        function x = x(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.xyz)
                error('xyz must be set to a value');
            end
            x = obj.xyz(idx,1);
        end
        function y = y(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.xyz)
                error('xyz must be set to a value');
            end
            y = obj.xyz(idx,2);
        end
        function z = z(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.xyz)
                error('xyz must be set to a value');
            end
            z = obj.xyz(idx,3);
        end
        function east = east(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.enu)
                error('enu must be set to a value');
            end
            east = obj.enu(idx,1);
        end
        function north = north(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.enu)
                error('enu must be set to a value');
            end
            north = obj.enu(idx,2);
        end
        function up = up(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.enu)
                error('enu must be set to a value');
            end
            up = obj.enu(idx,3);
        end

        %% plot
        function plot(obj, idx)
            arguments
                obj gt.Gpos
                idx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if ~isempty(obj.enu)
                enu_ = obj.enu(idx,:);
            else
                % first position is origin
                llh_ = obj.llh(idx,:);
                enu_ = rtklib.llh2enu(llh_, llh_(1,:));
            end

            figure;
            tiledlayout(3,1,'TileSpacing','Compact');
            nexttile(1, [2 1]);
            plot(enu_(:,1), enu_(:,2), '.-');
            xlabel('East (m)');
            ylabel('North (m)');
            grid on; axis equal;
            nexttile;
            plot(enu_(:,3),'.-');
            grid on;
            ylabel('Up (m)');
            drawnow
        end

        %% help
        function help(~)
            doc gt.Gpos
        end

        %% overload
        function gerr = minus(obj, gpos)
            arguments
                obj gt.Gpos
                gpos gt.Gpos
            end
            gerr = obj.difference(gpos);
        end
    end

    methods(Access=private)
        function c = insertdata(~,a,idx,b)
            c = [a(1:size(a,1)<idx,:); b; a(1:size(a,1)>=idx,:)];
        end
    end
end