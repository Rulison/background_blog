;;; shadows.lisp
;;; Contains statistical definitions for mixture models, plus
;;; EM code of various kinds.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Definitions for mixture models
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defstruct normal-mixture-model
  distrib            ;;; alist of (component . weight) pairs
  m                  ;;; number of components
  )

(defun normal-mixture-model-weight (model j)
  (cdr (nth j (normal-mixture-model-distrib model))))

(defun normal-mixture-model-component (model j)
  (car (nth j (normal-mixture-model-distrib model))))

(defun normal-mixture-model-datum-probs (x model)
  (mapcar #'(lambda (g.w)
	      (* (cdr g.w)
		 (normal-model-prob x (car g.w))))
	  (normal-mixture-model-distrib model)))

(defun normal-mixture-model-datum-weights (x model)
  (vnormalize (normal-mixture-model-datum-probs x model)))

(defun normal-mixture-model-datum-classification (x model)
  (let ((classprobs (normal-mixture-model-datum-weights x model)))
    (position (apply #'max classprobs) classprobs :test #'=)))

(defun set-normal-mixture-model-weight (model j w)
  (setf (cdr (nth j (normal-mixture-model-distrib model))) w))

(defun set-normal-mixture-model-mean (model j mean)
  (setf (normal-model-mean (normal-mixture-model-component model j)) mean))

(defun set-normal-mixture-model-variance (model j variance)
  (setf (normal-model-variance (normal-mixture-model-component model j)) variance))

(defstruct multinormal-mixture-model
  distrib            ;;; alist of (component . weight) pairs
  m                  ;;; number of components
  )

(defun multinormal-mixture-model-weight (model j)
  (cdr (nth j (multinormal-mixture-model-distrib model))))

(defun multinormal-mixture-model-component (model j)
  (car (nth j (multinormal-mixture-model-distrib model))))

(defun multinormal-mixture-model-datum-probs (x model)
  (mapcar #'(lambda (g.w)
	      (* (cdr g.w)
		 (multinormal-model-prob x (car g.w))))
	  (multinormal-mixture-model-distrib model)))

(defun multinormal-mixture-model-datum-weights (x model)
  (vnormalize (multinormal-mixture-model-datum-probs x model)))

(defun multinormal-mixture-model-datum-classification (x model)
  (let ((classprobs (multinormal-mixture-model-datum-weights x model)))
    (position (apply #'max classprobs) classprobs :test #'=)))

(defun set-multinormal-mixture-model-weight (model j w)
  (setf (cdr (nth j (multinormal-mixture-model-distrib model))) w))

(defun set-multinormal-mixture-model-mean (model j mean)
  (setf (multinormal-model-mean (multinormal-mixture-model-component model j)) mean))

(defun set-multinormal-mixture-model-covariance (model j covariance)
  (setf (multinormal-model-covariance 
         (multinormal-mixture-model-component model j)) covariance))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Functions for regular EM with univariate Gaussians
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun normal-mixture-EM (data m 
                          &optional (delta 0.00001) ;;; minimum relative change in LL
                                    (model (normal-mixture-k-means data m))
				    (max-iterations 1000000)
                          &aux (n (length data))
                               (weights (make-array (list n))) ;;; list for each datum
                               i (LL -1.0d6) (old-LL -1.0d7))
  (do ((iteration 1 (1+ iteration)))
      ((or (> iteration max-iterations)
	   (< (/ (- LL old-LL) (abs old-LL)) delta))
       model)
    (dprint (list "Current model has likelihood" LL))
    (dotimes (j m) 
      (dprint (list 'mean (normal-model-mean
			   (normal-mixture-model-component model j))
		    'sigma (sqrt (normal-model-variance
			   (normal-mixture-model-component model j)))
		    'weight (normal-mixture-model-weight model j))))
    (setq old-LL LL LL 0.0d0)

    ;;; first calculate probability raw p_ij of datum i in component j
    ;;; then normalized w_ij = w_jN_j(x_i) / \sum_j w_jN_j(x_i)

    (setq i 0)
    (dolist (x_i data)
      (let ((p_ijs (normal-mixture-model-datum-probs x_i model)))
	(incf LL (log (apply #'+ p_ijs)))
	(setf (aref weights i) (vnormalize p_ijs))
	(incf i)))

    ;;; now update the model
    ;;; w_j' = \sum_i p_ij / \sum_j \sum_i p_ij
    (dolist (g.w (normal-mixture-model-distrib model))
      (setf (cdr g.w) 0.0d0))
    (dotimes (i n) 
      (mapcar #'(lambda (g.w p_ij) (incf (cdr g.w) p_ij))
	      (normal-mixture-model-distrib model) (aref weights i)))
    (dnormalize (normal-mixture-model-distrib model))
    ;;; mu_j' = \sum_i p_ij x_i / \sum_i p_ij
    ;;; var_j' = (1/(N-1)) \sum_i p_ij (x_i - mu_j')^2 / \sum_i p_ij
    (dotimes (j m)
      (let* ((N_j (normal-mixture-model-component model j))
             (p_j 0.0d0) (mu_j 0.0d0) (var_j 0.0d0))
	(setq i 0)
        (dolist (x_i data)
	  (let ((p_ij (nth j (aref weights i))))
	    (incf p_j p_ij)
	    (incf mu_j (* p_ij x_i))
	    (incf i)))
        (setf mu_j (/ mu_j p_j))
        (setf (normal-model-mean N_j) mu_j)
	(setq i 0)
        (dolist (x_i data)
          (incf var_j (* (nth j (aref weights i))
			 (square (- x_i mu_j))))
	  (incf i))
        (setf var_j (/ var_j p_j))
        (setf (normal-model-variance N_j) var_j)))
    ))
    
;;; initialize with uniform weights
;;; initialize means at random in range
;;; initialize variances at (range/m)^2

(defun initialize-normal-mixture (data m)
  (let* ((min (apply #'min data))
         (range (- (apply #'max data) min))
         (model (make-normal-mixture-model :distrib (make-list m) :m m)))
    (dotimes (j m model)
      (setf (nth j (normal-mixture-model-distrib model))
            (cons (make-normal-model
                   :mean (+ min (random range))
                   :variance (square (/ range m)))
                  (float (/ 1 m)))))))

;;; testing

(defvar d1) (defvar d2) (defvar d3) (defvar m3) (defvar data3)
;;; the following seems a reasonable model for the data in pixel (240,120)
(defun imodel ()
  (setq d1 (make-normal-model :mean 92 :variance 144))
  (setq d2 (make-normal-model :mean 120 :variance 36))
  (setq d3 (make-normal-model :mean 160 :variance 1600))
  (setq m3 (make-normal-mixture-model :m 3
          :distrib (list (cons d1 0.38) (cons d2 0.37) (cons d3 0.25))))
  m3)

(imodel)
(setq data3-1000 nil)
(dotimes (i 1000) (push (random-from-normal-mixture m3) data3-1000))
(plot (frequency-counts data3-1000) "freq.data")

(defvar *debugging* nil)
(defun dprint (x) (when *debugging* (print x)) x)
(defun dprinc (x) (when *debugging* (princ x)) x)
                                                   
         
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Functions for incremental k-means with univariate Gaussians
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; identical to EM except that p_ij is 1 iff component j is the
;;; best explanation for datum i
;;; need to avoid using component weight in deciding this

(defun normal-mixture-k-means (data m 
                          &optional (delta 0.0001)
                          &aux (model (initialize-normal-mixture data m))
                               (n (length data))
                               (weights (make-array (list n m)))
                               (max (1+ delta))
                               x_i w_i p_ij)
  (do ()
      ((> delta max) model)
;    (dprint model)
    (setq max 0.0d0)
    ;;; first calculate probability p_ij of datum i in component j
    ;;; p_ij = w_jN_j(x_i) / \sum_j w_jN_j(x_i)
    (dotimes (i n)
      (setf x_i (nth i data) w_i 0.0d0)
      (let ((p* 0.0d0) (j* nil))
        (dotimes (j m)
          (setq p_ij (* (normal-mixture-model-weight model j) ;;; was 1.0
                        (normal-model-prob 
                         x_i (normal-mixture-model-component model j))))
          (when (> p_ij p*)
            (setf p* p_ij j* j))
          (incf w_i p_ij)
          (setf (aref weights i j) p_ij))
        (dotimes (j m)
          (setf (aref weights i j) (if (= j j*) 1.0d0 0.0d0)))))
;    (dprint weights)
    ;;; now update the model
    ;;; w_j' = \sum_i p_ij / \sum_j \sum_i p_ij
    (dotimes (j m)
      (let ((w_j 0.0d0))
        (dotimes (i n) (incf w_j (aref weights i j)))
        (set-normal-mixture-model-weight model j w_j)))
    (dnormalize (normal-mixture-model-distrib model))
    ;;; mu_j' = \sum_i p_ij x_i / \sum_i p_ij
    ;;; var_j' = (1/(N-1)) \sum_i p_ij (x_i - mu_j')^2 / \sum_i p_ij
    (dotimes (j m)
      (let* ((N_j (normal-mixture-model-component model j))
             (old-mu_j (normal-model-mean N_j))
             (old-var_j (normal-model-variance N_j))
             (p_j 0.0d0) (mu_j 0.0d0) (var_j 0.0d0))
        (dotimes (i n) 
          (incf p_j (aref weights i j))
          (incf mu_j (* (aref weights i j) (nth i data))))
        (setf mu_j (if (zerop p_j) 0.0d0 (/ mu_j p_j)))
        (setf max (max max (abs (- mu_j old-mu_j))))
;        (dprinc (list 'max max))
        (setf (normal-model-mean N_j) mu_j)
        (dotimes (i n)
          (incf var_j (* (aref weights i j) (square (- (nth i data) mu_j)))))
        (setf var_j (if (zerop p_j) 0.0d0 (/ var_j p_j)))
        (setf max (max max (abs (- var_j old-var_j))))
;        (dprinc (list 'max max))
        (setf (normal-model-variance N_j) var_j)))
    ))
          

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Functions for incremental EM with multivariate Gaussians
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defun multinormal-mixture-EM (data m 
                          &optional (delta 0.00001)
                                    (model (initialize-multinormal-mixture data m))
				    (max-iterations 1000000)
                          &aux (n (length data))
                               (d (when data (array-dimension (first data) 0)))
                               (weights (make-array (list n)))
                               i (LL -1.0d6) (old-LL -1.0d7))
  (do ((iteration 1 (1+ iteration)))
      ((or (> iteration max-iterations)
	   (< (/ (- LL old-LL) (abs old-LL)) delta))
       model)
    (dprint (list "Current model has likelihood" LL))
    (dotimes (j m) 
      (dprint (list 'mean (multinormal-model-mean
			   (multinormal-mixture-model-component model j))
		    'sigma (sqrt (determinant (multinormal-model-covariance
			     (multinormal-mixture-model-component model j))))
		    'weight (multinormal-mixture-model-weight model j))))
    (setq old-LL LL LL 0.0d0)

    ;;; first calculate probability p_ij of datum i in component j
    ;;; p_ij = w_jN_j(x_i) / \sum_j w_jN_j(x_i)

    (setq i 0)
    (dolist (x_i data)
      (let ((p_ijs (multinormal-mixture-model-datum-probs x_i model)))
	(incf LL (log (apply #'+ p_ijs)))
	(setf (aref weights i) (vnormalize p_ijs))
	(incf i)))

    ;;; now update the model
    ;;; w_j' = \sum_i p_ij / \sum_j \sum_i p_ij
    (dolist (g.w (multinormal-mixture-model-distrib model))
      (setf (cdr g.w) 0.0d0))
    (dotimes (i n) 
      (mapcar #'(lambda (g.w p_ij) (incf (cdr g.w) p_ij))
	      (multinormal-mixture-model-distrib model) (aref weights i)))
    (dnormalize (normal-mixture-model-distrib model))
    ;;; mu_j' = \sum_i p_ij x_i / \sum_i p_ij
    ;;; covar_j' = (1/(N-1)) \sum_i p_ij (x_i - mu_j')^2 / \sum_i p_ij
    (dotimes (j m)
      (let* ((N_j (multinormal-mixture-model-component model j))
             (p_j 0.0d0) 
             (mu_j (make-array (list d 1) :initial-element 0.0d0))
             (covar_j (make-array (list d d) :initial-element 0.0d0)))
	(setq i 0)
        (dolist (x_i data)
	  (let ((p_ij (nth j (aref weights i))))
	    (incf p_j p_ij)
	    (dotimes (k d)
              (incf (aref mu_j k 0)
                    (* p_ij (aref x_i k 0)))))
	  (incf i))
        (dotimes (k d) (setf (aref mu_j k 0) (/ (aref mu_j k 0) p_j)))
        (setf (multinormal-model-mean N_j) mu_j)
	(setq i 0)
        (dolist (x_i data)
          (dotimes (k d)
            (dotimes (l d)
              (incf (aref covar_j k l)
                    (* (nth j (aref weights i))
                       (- (aref x_i k 0)
                          (aref mu_j k 0))
                       (- (aref x_i l 0)
                          (aref mu_j l 0))))))
	  (incf i))
        (dotimes (k d)
          (dotimes (l d)
            (setf (aref covar_j k l) (/ (aref covar_j k l) p_j))))
        (setf (multinormal-model-covariance N_j) covar_j)))
    ))
    
;;; initialize with uniform weights
;;; initialize means at random in range
;;; initialize variances at (range/m)^2

(defun initialize-multinormal-mixture (data m
                                  &aux (d (array-dimension (first data) 0))
                                       (min (make-array (list d 1)))
                                       (range (make-array (list d 1)))
                                       (model (make-multinormal-mixture-model 
                                               :distrib (make-list m) :m m)))
  (dotimes (k d)
    (setf (aref min k 0)
          (apply #'min (mapcar #'(lambda (datum) (aref datum k 0)) data)))
    (setf (aref range k 0)
          (- (apply #'max (mapcar #'(lambda (datum) (aref datum k 0)) data))
             (aref min k 0))))
         
  (dotimes (j m model)
    (let ((mean (make-array (list d 1)))
          (covariance (make-array (list d d) :initial-element 0.0d0)))
      (setf (nth j (multinormal-mixture-model-distrib model))
            (cons (make-multinormal-model
                   :mean (dotimes (k d mean)
                           (setf (aref mean k 0)
                                 (+ (aref min k 0) (random (aref range k 0)))))
                   :covariance (dotimes (k d covariance)
                                 (setf (aref covariance k k)
                                       (square (/ (aref range k 0)
                                                  (* 2 (expt m (/ 1 d))))))))
                  (float (/ 1 m)))))))

;;; testing

(defvar md1) (defvar md2) (defvar md3) (defvar mm3) (defvar mdata3)
(defun rgbmodel ()
  ;;; shadow
  (setq md1 (make-multinormal-model 
	     :mean (make-array '(3 1) :initial-contents '((120) (100) (100)))
	     :covariance (make-array '(3 3) :initial-contents 
                           '((400 256 144)
			     (256 256 196)
			     (144 196 324)))))
  ;;; sunlit road
  (setq md2 (make-multinormal-model 
	     :mean (make-array '(3 1) :initial-contents '((140) (125) (160)))
	     :covariance (make-array '(3 3) :initial-contents 
                           '((40  25  9)
			     (25  25  16)
			     (9   16  20)))))
  ;;; car
  (setq md3 (make-multinormal-model 
	     :mean (make-array '(3 1) :initial-contents '((180) (170) (170)))
	     :covariance (make-array '(3 3) :initial-contents 
                           '((1600 1600 900)
			     (1600 2500 2000)
			     (900 2000 3000)))))
  ;;; mixture 35% shadow, 40% road, 25% car
  (setq mm3 (make-multinormal-mixture-model 
	     :m 3
	     :distrib (list (cons md1 0.35) (cons md2 0.40) (cons md3 0.25))))
  mm3)

;(setq mdata3-5000 nil)
;(dotimes (i 5000) (push (random-from-multinormal-mixture mm3) mdata3-5000))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; code for constructing models for a whole video
;;; pretty cruddy for now

(defvar *i-models* (make-array '(320 240)))

(defun extract-models (root start finish extension xmin xmax ymin ymax)
  (loop for x from xmin to xmax do
    (loop for y from ymin to ymax do
      (setq *rgb* nil) 
      (do-video #'(lambda (file) (extract-pixel x y file)) 
		root start finish extension)
      (setf (aref *i-models* x y)
	    (normal-mixture-EM (mapcar #'rgb->intensity *rgb*) 3 0.01 m3)) 
      )))
