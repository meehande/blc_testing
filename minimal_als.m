function  [results] = minimal_als(varargin)
% use maxiter to compare with Deirdre

    offset = [];
    p = 1; %number of groups, starts with 1
    n = 15; %number of users
    m = 20; % number of items
    d = 10; %number of features
    sigma2Utilde = 1/eps; %variance for Utilde
    sigma2V = 1/eps; %variance for V
    sigmaP = 1/eps;%1/eps;
    sigmanetflix = [];
    sigmaprivate = [];
    tolerance = 10^-4; %alsqr tolerance
    tolerance_mainloop = [];
    maxiter = 1000; % max gradient descent or alsqr iterations
    maxiter_mainloop = [];
    verbose = 0;
    fast = false;
    missing_ratio = [];
    frequency = 100;
    tolerance_netflix=[];
    maxiter_netflix=[];
    range = [];
    final_groups = 16; %max doubling
    n_tries = []; %number of doubling it overrides final_groups
    visited = [];
    packed = false;
    octave = false;
    tosave = false;
    seed = [];
    test_values = [];
    name = [];
    keep_doubling = false; %keep doubling nyms even if it's making performance worst
    gamma = []; %[0 1 10 1000]; % this is the parameter for the least squares ratio between group and self-profiling
    % this is just to create variable scope please ignore!
    P = [];Utilde = [];V = [];Q = [];Rtilde = [];R = [];Lambdavector=[];
    recover = false;
    funcParamsNames = {'final_groups','R', 'n', 'm','d','sigma2','sigma2Utilde','sigma2V','sigmanetflix','tolerance','maxiter','n_tries','maxiter_mainloop','tolerance_mainloop','verbose','missing_ratio','frequency','offset','fast','seed','packed','sigmaprivate','gamma','tolerance_netflix','maxiter_netflix','octave','test_values','range','tosave','keep_doubling','name','P','Utilde','V','recover','p'};
    assignUserInputs(funcParamsNames,varargin{:});
    if ~isempty(seed)
        if octave
            rand('seed',seed)
        else
            rng(seed);
        end
    end
    if isempty(n_tries)
        n_tries = ceil(log2(final_groups))+1;
    end

    if octave
        do_braindead_shortcircuit_evaluation(1);
        initial_seed = rand('seed');
    else
        initial_seed = rng;
    end
        

    if isempty(tolerance_netflix)
    tolerance_netflix = tolerance;
    end

    if isempty(maxiter_mainloop)
    maxiter_mainloop = maxiter;
    end

    if isempty(tolerance_mainloop)
    tolerance_mainloop = tolerance;
    end

    if isempty(maxiter_netflix)
    maxiter_netflix = maxiter;
    end

    if isempty(sigmanetflix)
       sigmanetflix = sigma2V;
    end

    if isempty(sigmaprivate)
        sigmaprivate = sigma2V;
    end

    if isempty(R)
        error('I need R')
    else
        [n,m] = size(R);
    end

    if fast
        warning off
    end

    if issparse(R)
        worksparse = true;
    else
        worksparse = false;
        if any(~isfinite(R(:)))
            warning('legacy format: trying to convert missing values from nan to zeros')
            if any(R(:)==0)
                error('can''t convert: there are missing values equal to zero too')
            end
            R(isnan(R(:))) = 0;
        end
    end

    tR = R;
    existingvaluestR = logical(spones(tR));
    totalvaluestR = nnz(tR);
    if ~isempty(missing_ratio) %remove some values randomly, preserving some rows if we want
        n_toremove = ceil((totalvaluestR)*missing_ratio);
        if verbose; disp(['We will remove ' num2str(n_toremove) ' random elements from R']);end
        toremove = randperm(totalvaluestR,n_toremove);
        temp = find(existingvaluestR);
        R(temp(toremove)) = 0;
        clear temp;
        clear toremove;


    %% clean tR    
        I = spones(R);
        rows_to_remove = ~any(I');
        tR(rows_to_remove,:) = [];
        clear rows_to_remove;
        columns_to_remove = ~any(I);
        tR(:,columns_to_remove) = [];
        clear columns_to_remove;
        [n,m] = size(tR);
        existingvaluestR = logical(spones(tR));

    %% clean R
        t = double(I(any(I'),any(I)));
        [i2,j2,~] = find(t);
        [~,~,s] = find(R);
        R = sparse(i2,j2,s);
        clear t i2 j2 s;

    end

    existingvalues = logical(spones(R)); %this gives us U(v) and V(u)
    existingvaluestilde = [];
    totalvalues = nnz(R);
    is_item_observed = any(existingvalues,1); % whether a item is used
    item_observed = find(is_item_observed); % whether a item is used
    if isempty(test_values)
        if worksparse
            test_values = tR;
            test_values(existingvalues) = 0;
        else
            test_values = (tR .* (1 - existingvalues));
        end
    end
    test_total = nnz(test_values);
    existing_test = logical(spones(test_values));

    if isempty(offset)
        offset = mean(tR(existingvaluestR(:)));
    end
        
    % FIX frequency
    frequency = min(frequency,n);

    % variables of the system - random initial values
    if ~recover
        Utilde = randn(d,p) +  sign(offset)*sqrt(abs(offset)/d);V= randn(d,m) + sign(offset)*sqrt(abs(offset)/d);
    else
        [d,p] = size(Utilde);
    end


    % cached values
    UUU = eye(d)./sigma2Utilde;
    VVV = eye(d)./sigma2V;
    VVVnetflix = eye(d)./sigmanetflix;
    UUUnetflix = eye(d)./sigmanetflix;
    VVVprivate = eye(d)./sigmaprivate;

    if tosave
        [stat,filename] = unix(['mktemp --tmpdir="$PWD" --suffix=' name '.mat']);
        if stat == 0
            [stat,filename]=unix(['basename "' filename(1:end-1)  '"']);
        end
        if stat==0
            filename = [filename(1:end-1)];
        else
            filename = 'temptree.mat';
        end
    end 

    clear todebug;
    todebug;
    if worksparse
        if ~recover;P=sparse(p,n,false);end;
    else
        if ~recover;P = false(p,n);end;
    end
    if recover;expand_tree();end
    temp = repmat(eye(p),1,ceil(n/p));
    %P = temp(:,1:n); % Generalized identity
    updateP(temp(:,1:n))
    
    UV = Utilde'*V;
    start_time = tic;
    alternating_lsqr();
    results.tim = toc(start_time)
    UV = Utilde'*V;
%    results.er = prediction_error();
    results.U = Utilde;
    results.V = V;


function updateP(newP) % update all the other variables when we change P (note that this doesn't change Utilde nor V)
    P = newP;
    Rhat = zeros(p,m);
    Rtilde = nan(p,m);
    Lambdavector = zeros(p,m);
    for v = item_observed
        Lambdavector(:,v) = sum(P(:,(existingvalues(:,v))),2);
        Rhat(:,v) = sum(repmat(R(existingvalues(:,v),v),1,p)'.*P(:,existingvalues(:,v)),2); % ** speed up this?
        iL =  diag(Lambdavector(:,v));
        iL(iL>0) = iL(iL>0).^-1;
        Rtilde(:,v) = iL*Rhat(:,v); % Done: speed up this by doing inverse manually!
    end
    existingvaluestilde = Lambdavector>0;
end

function alternating_lsqr()
    total = nnz(existingvaluestilde);
    if total == 0
        error('cacca')
    end
    for i=1:maxiter
        for g=1:p
            temp = Lambdavector(g,existingvaluestilde(g,:))';
            ltemp = numel(temp);
            if sum(temp)
                if fast
%                    Utilde(:,g) =  Rtilde(g,existingvaluestilde(g,:))*spdiags(temp,0,ltemp,ltemp)*V(:,(existingvaluestilde(g,:)))'/(VVV+V(:,(existingvaluestilde(g,:)))*spdiags(temp,0,ltemp,ltemp)*V(:,(existingvaluestilde(g,:)))');
                    Utilde(:,g) =  Rtilde(g,existingvaluestilde(g,:))*diag(temp)*V(:,(existingvaluestilde(g,:)))'/(VVV+V(:,(existingvaluestilde(g,:)))*diag(temp)*V(:,(existingvaluestilde(g,:)))');
                    
                else
%                    Utilde(:,g) =  Rtilde(g,existingvaluestilde(g,:))*spdiags(temp,0,ltemp,ltemp)*V(:,(existingvaluestilde(g,:)))'*pinv(VVV+V(:,(existingvaluestilde(g,:)))*spdiags(temp,0,ltemp,ltemp)*V(:,(existingvaluestilde(g,:)))');
                    Utilde(:,g) =  Rtilde(g,existingvaluestilde(g,:))*diag(temp)*V(:,(existingvaluestilde(g,:)))'*pinv(VVV+V(:,(existingvaluestilde(g,:)))*diag(temp)*V(:,(existingvaluestilde(g,:)))');
                end

            end
            
        end
        for v=item_observed
            if sum(Lambdavector(:,v))
                if fast
                    V(:,v) = Rtilde(:,v)'* diag(Lambdavector(:,v))*Utilde'/(UUU + Utilde* diag(Lambdavector(:,v))*Utilde');
                else
                    V(:,v) = Rtilde(:,v)'*diag(Lambdavector(:,v))*Utilde'*pinv(UUU + Utilde* diag(Lambdavector(:,v))*Utilde');
                end
            end
        end
        temp = whos
        results.mem = sum([temp.bytes])
%        UV = Utilde'*V;
%        er = sqrt(sum(nansum((Rtilde(existingvaluestilde) - UV(existingvaluestilde)).^2))/total);
%        if er<=tolerance || abs(er-preverror)<=tolerance; return;  end
%       preverror = er;
    end
end

function assignUserInputs(funcParamsNames, varargin)

%% Load uses params, overifding default ones

    % The list of all legal input parameter names. Others will be ignored
    if ~(iscell(funcParamsNames) && all( cellfun(@ischar, funcParamsNames) ))
        % if no cell array of string was specified as funcParamsNames input, consider it to
        % be empty, and append varargin
        varargin=cat(2, funcParamsNames, varargin); % varargin is a 1xN cell array
        funcParamsNames=[];
    end

    if numel(varargin)==1 && isempty(varargin{1}) % functioned called wihout arguments
        return;
    end

    % verify if no funcParamsNames input was specified
    if isempty(funcParamsNames)
        isNoFuncParamsNames=true;
    else
        [~, iOrigA, ~]=unique(funcParamsNames);
        if length(iOrigA)<length(funcParamsNames) 
            % remove repeating elements, keeping same ordering
            funcParamsNames=funcParamsNames( sort(iOrigA) );
        end
        isNoFuncParamsNames=false;
    end

    unUsedVarargin=varargin;
    isUsedParamsName=false( size(funcParamsNames) );
    %% Load "param name"- "param value" pairs.
    nVarargin=length(varargin);
    if nVarargin > 1
        isSpecificParam=false(1, nVarargin);
        iArg=1;
        while iArg <= (nVarargin-1)
            % automatically get all RELEVANT input pairs and store in local vairables
            isFuncParamsNames=strcmpi(varargin{iArg}, funcParamsNames);
            if isNoFuncParamsNames || any( isFuncParamsNames  )
                assignin('caller', varargin{iArg}, varargin{iArg+1});
                
                isSpecificParam( [iArg, iArg+1] )=true; % parameters come in Name-Value pairs
                iArg=iArg+1;
                isUsedParamsName(isFuncParamsNames)=true;
            end
            iArg=iArg+1;
        end % while iArg < (nVarargin-1)
        unUsedVarargin=varargin(~isSpecificParam); % Save varargin elements that were not used
        funcParamsNames=funcParamsNames(~isUsedParamsName); % do not allow repeating 
                                                            % initialisation of variables
        isUsedParamsName=false( size(funcParamsNames) );                                                 
    end % if nargin>1

    %% Attempt loading users parameters from input structures.
    % In this case structure field name will be parameter name, and filed content will be
    % parameter value.

    iStructures=find( cellfun(@isstruct, unUsedVarargin) );
    if ~isempty(iStructures)
        isSpecificParam=false(iStructures);
    end
    for iStruct=iStructures
        % analyze each structure unattained by previous "Load param name- param value pairs"
        % process
        CurrStruct=unUsedVarargin{iStruct};
        if numel(CurrStruct)>1
            CurrStruct=CurrStruct(1);
            warning('Structure arrays are not supported, 1''st element will be used.');
        elseif isempty(CurrStruct) % Ignore empty structures
            continue;
        end
        currFieldNames=fieldnames(CurrStruct);
        if isNoFuncParamsNames
            funcParamsNames=currFieldNames;
        end

        nFields=length(currFieldNames);
        for iFieldStr=1:nFields
            % Find relevant structure field names supported by function legal input names
            isFuncParamsNames=strcmpi(currFieldNames{iFieldStr}, funcParamsNames);
            if sum(isFuncParamsNames) > 1 % if several names were found try case sensitive match
                isFuncParamsNames=strcmp(currFieldNames{iFieldStr}, funcParamsNames);
            end
            
            % in case of ambiguty, use the first appearing name, as they are identical.
            iFirstFittingName=find(isFuncParamsNames, 1, 'first'); 
            
            if ~isempty(iFirstFittingName) % Load parameters into current environment
                assignin('caller',  funcParamsNames{iFirstFittingName},...
                    CurrStruct.(currFieldNames{iFieldStr}) );
                isSpecificParam(iStruct)=true; % mark used input
                isUsedParamsName(iFirstFittingName)=true;
            end
        end % for iFieldStr=1:nFields
        if isNoFuncParamsNames
            funcParamsNames=[];
        else
            funcParamsNames=funcParamsNames(~isUsedParamsName); % do not allow repeating
                                                            % initialisation of variables
        end
    end % for iStruct=find( cellfun(@isstruct, unUsedVarargin) )


    if ~isempty(iStructures)
        % remove used input elements
        unUsedVarargin=unUsedVarargin( iStructures(~isSpecificParam) ); 
    end
    if isequal(unUsedVarargin, varargin) % neither inputs were used to extract user inputs
        % Preserve custom Matlab input parameters transfer scheme. Here inpus order defines
        % the variable destination.
        nInputs=min( numel(varargin), numel(funcParamsNames) );
        for iVargin=1:nInputs
            assignin( 'caller',  funcParamsNames{iVargin}, varargin{iVargin} );
        end
    end
end


function y = nansum(x,dim)
	% FORMAT: Y = NANSUM(X,DIM)
	% 
	%    Sum of values ignoring NaNs
	%
	%    This function enhances the functionality of NANSUM as distributed in
	%    the MATLAB Statistics Toolbox and is meant as a replacement (hence the
	%    identical name).  
	%
	%    NANSUM(X,DIM) calculates the mean along any dimension of the N-D array
	%    X ignoring NaNs.  If DIM is omitted NANSUM averages along the first
	%    non-singleton dimension of X.
	%
	%    Similar replacements exist for NANMEAN, NANSTD, NANMEDIAN, NANMIN, and
	%    NANMAX which are all part of the NaN-suite.
	%
	%    See also SUM

	% -------------------------------------------------------------------------
	%    author:      Jan Gläscher
	%    affiliation: Neuroimage Nord, University of Hamburg, Germany
	%    email:       glaescher@uke.uni-hamburg.de
	%    
	%    $Revision: 1.2 $ $Date: 2005/06/13 12:14:38 $

	if isempty(x)
		y = [];
		return
	end

	if nargin < 2
		dim = min(find(size(x)~=1));
		if isempty(dim)
			dim = 1;
		end
	end

	% Replace NaNs with zeros.
	nans = isnan(x);
	x(isnan(x)) = 0; 

	% Protect against all NaNs in one dimension
	count = size(x,dim) - sum(nans,dim);
	i = find(count==0);

	y = sum(x,dim);
	y(i) = NaN;

end

function [err] = prediction_error()
if test_total
        try
            PUV = P'*UV; % if this is too big we will do item by item
            er = sum((paren(test_values(existing_test),':') - paren(PUV(existing_test),':')).^2);
            err = sqrt(er/test_total);
        catch
            err = 0;
             err = zeros(1,n);
	     for i=1:n
               err(i) = sum((test_values(i,existing_test(i,:)) - UV(P(:,i),existing_test(i,:))   ).^2);
             end
	     err = sum(err);
             err = sqrt(err/test_total);
        end
else
err = 0
end
end

function out =  paren(x, varargin)
% allow to index the output of another function
    out = x(varargin{:});
end



end
